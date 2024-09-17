from tokenizers import Tokenizer
from smart_open import open
from tqdm import tqdm
import textstat
from operator import itemgetter

import argparse

parser = argparse.ArgumentParser(description="Cached Dataset Creation")
parser.add_argument(
    "--segments_path",
    type=str,
    default="/work/vesuvius/babylm/data/processed/segmented.txt",
    help="Path to the segmented data file.",
)
parser.add_argument(
    "--tokenizer_path",
    type=str,
    default="/work/vesuvius/babylm/models/StructELC_10M_16k_ordered/tokenizer.json",
    help="Path to the tokenizer JSON file.",
)
parser.add_argument(
    "--sequence_length",
    type=int,
    default=128,
    help="Sequence length of each cached input sequence.",
)
args = parser.parse_args()


SEQ_LEN = args.sequence_length - 2
tokenizer = Tokenizer.from_file(args.tokenizer_path)


documents = [[]]
for line in tqdm(open(args.segments_path)):
    line = line.strip()

    if len(line) == 0:
        if len(documents[-1]) > 0:
            documents.append([])
        continue

    ids = tokenizer.encode(line, add_special_tokens=False).ids
    documents[-1].append(ids)

output = []

for document in tqdm(documents):
    segment = []
    for i, sentence in enumerate(document):
        segment += sentence

        if len(segment) > SEQ_LEN:
            segment = segment[:SEQ_LEN]
            subwords = [tokenizer.id_to_token(token_id) for token_id in segment]
            output.append(" ".join(subwords))
            segment = [s for s in sentence]

    if len(segment) > 0:
        segment = segment[:SEQ_LEN]
        subwords = [tokenizer.id_to_token(token_id) for token_id in segment]
        output.append(" ".join(subwords))


def assign_score(token_str):
    tokens = token_str.split(" ")
    ids = [tokenizer.token_to_id(token) for token in tokens]
    return textstat.flesch_reading_ease(tokenizer.decode(ids))


ranked_segments = [(segment, assign_score(segment)) for segment in output]

# sort the decoded segments by how difficult they are to read
ranked_segments = sorted(ranked_segments, key=itemgetter(1), reverse=True)


with open(f"../../../data/processed/cached_{SEQ_LEN + 2}.txt", "w") as f:
    for segment_text in ranked_segments:
        f.write(segment_text[0] + "\n")
