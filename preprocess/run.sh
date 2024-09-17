mkdir -p ../../text_data/processed

python3 bnc_spoken.py
python3 gutenberg.py
python3 open_subtitles.py
python3 simple_wikipedia.py
python3 switchboard.py
python3 childes.py


cat ../../../data/processed/bnc_spoken.txt  ../../../data/processed/gutenberg.txt ../../../data/processed/open_subtitles.txt  ../../../data/processed/simple_wikipedia.txt ../../../data/processed/switchboard.txt ../../../data/processed/childes.txt > ../../../data/processed/all.txt

python3 segment.py