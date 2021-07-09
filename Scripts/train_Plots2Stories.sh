
python train.py \
Plots2Stories \
"Data/Plots2Stories/src_vocab.pt" \
--TGT_VOCAB "Data/Plots2Stories/tgt_vocab.pt" \
"Data/Plots2Stories/train_data.pt" \
"Data/Plots2Stories/valid_data.pt" \
"Data/Plots2Stories/test_data.pt" \
--NUM_ENCODER_LAYERS 12 \
--NUM_DECODER_LAYERS 12 \
--EMB_SIZE 768 \
--FFN_HID_DIM 768 \
--DROPOUT 0.1 \
--NHEAD 8 \
--BATCH_SIZE 100 \
--NUM_EPOCHS 16


