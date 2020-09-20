# obtain initial file
cd data
#dvc run -n extract-input-zip \
#        -d pix512.zip \
#        -o pix512 \
#         unzip -qq  pix512.zip

cp -r pix512 train_stage
cp -r pix512 test_stage

cd ..
python src/generate-augmentation-for-train-test.py



