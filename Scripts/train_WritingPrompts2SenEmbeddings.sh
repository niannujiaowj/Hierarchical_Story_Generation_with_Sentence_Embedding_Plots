python Scripts/train.py \
WritingPrompts2SenEmbeddings \
"Data/WritingPrompts2SenEmbeddings/src_vocab.pt" \
"Data/WritingPrompts2SenEmbeddings/tgt_vocab.pt" \
"Data/WritingPrompts2SenEmbeddings/train_data.pt" \
"Data/WritingPrompts2SenEmbeddings/valid_data.pt" \
"Data/WritingPrompts2SenEmbeddings/test_data.pt" \
--BATCH_SIZE 512 \
--NUM_EPOCHS 16

