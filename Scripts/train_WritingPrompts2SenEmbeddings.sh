python train.py \
WritingPrompts2SenEmbeddings \
"Data/WritingPrompts2SenEmbeddings/src_vocab.pt" \
"Data/WritingPrompts2SenEmbeddings/tgt_vocab.pt" \
"Data/WritingPrompts2SenEmbeddings/train_data.pt" \
"Data/WritingPrompts2SenEmbeddings/valid_data.pt" \
"Data/WritingPrompts2SenEmbeddings/test_data.pt" \
--NUM_ENCODER_LAYERS 12 \
--NUM_DECODER_LAYERS 12 \
--EMB_SIZE 768 \
--FFN_HID_DIM 768 \
--DROPOUT 0.1 \
--NHEAD 8 \
--BATCH_SIZE 100 \
--NUM_EPOCHS 16


