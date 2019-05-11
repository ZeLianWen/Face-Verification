# Face-Verification

Method and Step:
1. Run process_image.m for reading training and test data.The result is AR_face_image_train.mat and AR_face_image_test.mat.
2. Run match_AR.m for generating test and training data.The result is AR_face_data_test.mat and AR_face_data_train.mat.
3. cnn.mat is a well-trained model, so you can use it directly by setting use_pre=true in cnn_examples.m.
4. Run cnn_examples.m.
