#!/usr/bin/env python
#
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare a corpus for processing by swivel.

Creates a sharded word co-occurrence matrix from a text file input corpus.

Usage:

  prep.py --output_dir <output-dir> --input <text-file>

Options:

  --input <filename>
      The input text.

  --output_dir <directory>
      Specifies the output directory where the various Swivel data
      files should be placed.

  --shard_size <int>
      Specifies the shard size; default 4096.

  --min_count <int>
      Specifies the minimum number of times a word should appear
      to be included in the vocabulary; default 5.

  --max_vocab <int>
      Specifies the maximum vocabulary size; default shard size
      times 1024.

  --vocab <filename>
      Use the specified unigram vocabulary instead of generating
      it from the corpus.

  --bufsz <int>
      The number of co-occurrences that are buffered; default 16M.

"""

import os
import csv
import struct
import sys

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('input', '', 'The input text.')
flags.DEFINE_string('output_dir', '/tmp/swivel_data',
                    'Output directory for Swivel data')
flags.DEFINE_integer('row_idx', 0, '0-based index for which field in input '
                     'contains row tokens')
flags.DEFINE_integer('col_idx', 1, '0-based index for which field in input '
                     'contains column tokens')
flags.DEFINE_integer('count_idx', 2, '0-based index for which field in input '
                     'contains counts. If < 0, assume no field and each line '
                     'is a single count.')
flags.DEFINE_integer('shard_size', 4096, 'The size for each shard')
flags.DEFINE_integer('min_count', 5,
                     'The minimum number of times a word should occur to be '
                     'included in the vocabulary')
flags.DEFINE_integer('max_vocab', 4096 * 64, 'The maximum vocabulary size')
flags.DEFINE_string('row_vocab', '', 'Row vocabulary to use instead of generating one')
flags.DEFINE_string('col_vocab', '', 'Column vocabulary to use instead of generating one')
flags.DEFINE_integer('bufsz', 16 * 1024 * 1024,
                     'The number of co-occurrences to buffer')

FLAGS = flags.FLAGS

shard_cooc_fmt = struct.Struct('iif')


def create_vocabulary(lines):
  """Reads text lines and generates a vocabulary."""
  row_vocab = {}
  col_vocab = {}
  csvreader = csv.reader(lines)
  for line in csvreader:
    row_tok = line[FLAGS.row_idx]
    col_tok = line[FLAGS.col_idx]
    row_vocab.setdefault(row_tok, 0)
    col_vocab.setdefault(col_tok, 0)
    if FLAGS.count_idx > 0:
      count = float(line[FLAGS.count_idx])
      row_vocab[row_tok] += count
      col_vocab[col_tok] += count
    else:
      row_vocab[row_tok] += 1
      col_vocab[col_tok] += 1

  row_vocab = [(tok, n) for tok, n in row_vocab.items() if n >= FLAGS.min_count]
  row_vocab.sort(key=lambda kv: (-kv[1], kv[0]))
  col_vocab = [(tok, n) for tok, n in col_vocab.items() if n >= FLAGS.min_count]
  col_vocab.sort(key=lambda kv: (-kv[1], kv[0]))

  num_row_toks = min(len(row_vocab), FLAGS.max_vocab)
  num_col_toks = min(len(col_vocab), FLAGS.max_vocab)
  if num_row_toks % FLAGS.shard_size != 0:
    #num_row_toks -= num_row_toks % FLAGS.shard_size
    num_row_toks += FLAGS.shard_size - (num_row_toks % FLAGS.shard_size)
  if num_col_toks % FLAGS.shard_size != 0:
    #num_col_toks -= num_col_toks % FLAGS.shard_size
    num_col_toks += FLAGS.shard_size - (num_col_toks % FLAGS.shard_size)

  if not num_row_toks:
    raise Exception('empty row vocabulary')
  if not num_col_toks:
    raise Exception('empty column vocabulary')

  print('row vocabulary contains %d tokens (original %d)' % (num_row_toks, len(row_vocab)))
  print('column vocabulary contains %d tokens (original %d)' % (num_col_toks, len(col_vocab)))

  #row_vocab = row_vocab[:num_row_toks]
  for i in range(num_row_toks - len(row_vocab)):
    row_vocab.append(('ROWPADDING_{0}'.format(i) , 0))
  #col_vocab = col_vocab[:num_col_toks]
  for i in range(num_col_toks - len(col_vocab)):
    col_vocab.append(('COLPADDING_{0}'.format(i) , 0))
  return [tok for tok, n in row_vocab], [tok for tok, n in col_vocab]


def write_vocab_and_sums(vocab, sums, vocab_filename, sums_filename):
  """Writes vocabulary and marginal sum files."""
  with open(os.path.join(FLAGS.output_dir, vocab_filename), 'w') as vocab_out:
    with open(os.path.join(FLAGS.output_dir, sums_filename), 'w') as sums_out:
      for tok, cnt in zip(vocab, sums):
        print(tok, file=vocab_out)
        print(cnt, file=sums_out)


def compute_coocs(lines, row_vocab, col_vocab):
  """Compute the co-occurrence statistics from the text.

  This generates a temporary file for each shard that contains the intermediate
  counts from the shard: these counts must be subsequently sorted and collated.

  """
  row_word_to_id = {tok: idx for idx, tok in enumerate(row_vocab)}
  col_word_to_id = {tok: idx for idx, tok in enumerate(col_vocab)}

  r_num_shards = len(row_vocab) // FLAGS.shard_size
  c_num_shards = len(col_vocab) // FLAGS.shard_size

  shardfiles = {}
  for row in range(r_num_shards):
    for col in range(c_num_shards):
      filename = os.path.join(
          FLAGS.output_dir, 'shard-%03d-%03d.tmp' % (row, col))

      shardfiles[(row, col)] = open(filename, 'wb+')

  def flush_coocs():
    for (row_id, col_id), cnt in coocs.items():
      row_shard = row_id % r_num_shards
      row_off = row_id % FLAGS.shard_size
      col_shard = col_id % c_num_shards
      col_off = col_id % FLAGS.shard_size

      shardfiles[(row_shard, col_shard)].write(
          shard_cooc_fmt.pack(row_off, col_off, cnt))

  coocs = {}
  row_sums = [0.0] * len(row_vocab)
  col_sums = [0.0] * len(col_vocab)

  csvreader = csv.reader(lines)
  for lineno, line in enumerate(csvreader, start=1):
    rid = row_word_to_id.get(line[FLAGS.row_idx], None)
    cid = col_word_to_id.get(line[FLAGS.col_idx], None)
    if rid is None or cid is None:
      continue
    pair = (min(rid, cid), max(rid, cid))
    if FLAGS.count_idx > 0:
      count = float(line[FLAGS.count_idx])
    else:
      count = 1
    row_sums[rid] += count
    col_sums[cid] += count
    coocs.setdefault(pair, 0.0)
    coocs[pair] += count

    if lineno % 10000 == 0:
      sys.stdout.write('\rComputing co-occurrences: %d lines processed...' % lineno)
      sys.stdout.flush()

      if len(coocs) > FLAGS.bufsz:
        flush_coocs()
        coocs = {}

  flush_coocs()
  sys.stdout.write('\n')

  return shardfiles, row_sums, col_sums


def write_shards(row_vocab, col_vocab, shardfiles):
  """Processes the temporary files to generate the final shard data.

  The shard data is stored as a tf.Example protos using a TFRecordWriter. The
  temporary files are removed from the filesystem once they've been processed.

  """
  num_row_shards = len(row_vocab) // FLAGS.shard_size
  num_col_shards = len(col_vocab) // FLAGS.shard_size

  ix = 0
  for (row, col), fh in shardfiles.items():
    ix += 1
    sys.stdout.write('\rwriting shard %d/%d' % (ix, len(shardfiles)))
    sys.stdout.flush()

    # Read the entire binary co-occurrence and unpack it into an array.
    fh.seek(0)
    buf = fh.read()
    os.unlink(fh.name)
    fh.close()

    coocs = [
        shard_cooc_fmt.unpack_from(buf, off)
        for off in range(0, len(buf), shard_cooc_fmt.size)]

    # Sort and merge co-occurrences for the same pairs.
    coocs.sort()

    if coocs:
      current_pos = 0
      current_row_col = (coocs[current_pos][0], coocs[current_pos][1])
      for next_pos in range(1, len(coocs)):
        next_row_col = (coocs[next_pos][0], coocs[next_pos][1])
        if current_row_col == next_row_col:
          coocs[current_pos] = (
              coocs[current_pos][0],
              coocs[current_pos][1],
              coocs[current_pos][2] + coocs[next_pos][2])
        else:
          current_pos += 1
          if current_pos < next_pos:
            coocs[current_pos] = coocs[next_pos]

          current_row_col = (coocs[current_pos][0], coocs[current_pos][1])

      coocs = coocs[:(1 + current_pos)]

    # Convert to a TF Example proto.
    def _int64s(xs):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(xs)))

    def _floats(xs):
      return tf.train.Feature(float_list=tf.train.FloatList(value=list(xs)))

    example = tf.train.Example(features=tf.train.Features(feature={
        'global_row': _int64s(
            row + num_row_shards * i for i in range(FLAGS.shard_size)),
        'global_col': _int64s(
            col + num_col_shards * i for i in range(FLAGS.shard_size)),

        'sparse_local_row': _int64s(cooc[0] for cooc in coocs),
        'sparse_local_col': _int64s(cooc[1] for cooc in coocs),
        'sparse_value': _floats(cooc[2] for cooc in coocs),
    }))

    filename = os.path.join(FLAGS.output_dir, 'shard-%03d-%03d.pb' % (row, col))
    with open(filename, 'wb') as out:
      out.write(example.SerializeToString())

  sys.stdout.write('\n')


def main(_):
  # Create the output directory, if necessary
  if FLAGS.output_dir and not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  # Read the file onces to create the vocabulary.
  if not FLAGS.row_vocab or not FLAGS.col_vocab:
    with open(FLAGS.input, 'r') as lines:
      row_vocab, col_vocab = create_vocabulary(lines)
  elif FLAGS.row_vocab:
    with open(FLAGS.row_vocab, 'r') as lines:
      row_vocab = [line.strip() for line in lines]
  elif FLAGS.col_vocab:
    with open(FLAGS.col_vocab, 'r') as lines:
      col_vocab = [line.strip() for line in lines]


  # Now read the file again to determine the co-occurrence stats.
  with open(FLAGS.input, 'r') as lines:
    shardfiles, row_sums, col_sums = compute_coocs(lines, row_vocab, col_vocab)

  # Collect individual shards into the shards.recs file.
  write_shards(row_vocab, col_vocab, shardfiles)

  # Now write the marginals.
  write_vocab_and_sums(row_vocab, row_sums, 'row_vocab.txt', 'row_sums.txt')
  write_vocab_and_sums(col_vocab, col_sums, 'col_vocab.txt', 'col_sums.txt')

  print('done!')


if __name__ == '__main__':
  tf.app.run()
