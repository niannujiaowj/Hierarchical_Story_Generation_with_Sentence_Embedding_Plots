python Scripts/preprocess.py \
Plots2Stories \
"Dataset/Plots/small" \
--TRAIN_SRC_SENINDICES "Data/WritingPrompts2SenEmbeddings/train_SenIndices.pt" \
"Dataset/WritingPrompts/small.wp_target" \
"Dataset/Plots/valid" \
--VALID_SRC_SENINDICES "Data/WritingPrompts2SenEmbeddings/valid_SenIndices.pt" \
"Dataset/WritingPrompts/valid.wp_target" \
"Dataset/Plots/test" \
--TEST_SRC_SENINDICES "Data/WritingPrompts2SenEmbeddings/test_SenIndices.pt" \
"Dataset/WritingPrompts/test.wp_target"

