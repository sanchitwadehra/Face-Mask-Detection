let w_mask = [];
let wo_mask = [];
let batchSize = 500;
let totalBatches = 8;
let currentBatch = 0;
let maskClassifier;

function preload() {
  // load the first batch of images
  loadBatch();
}

function loadBatch() {
  // determine the start and end index for the current batch
  let startIndex = currentBatch * batchSize + 1;
  let endIndex = startIndex + batchSize - 1;

  // load the w_mask images for the current batch
  for (let i = startIndex; i <= endIndex; i++) {
    if (i <= 3725) {
      w_mask.push(loadImage(`data/standardized/with_mask/with_mask_${i}_64x64.jpg`));
    }
  }

  // load the wo_mask images for the current batch
  for (let i = startIndex; i <= endIndex; i++) {
    if (i <= 3828) {
      wo_mask.push(loadImage(`data/standardized/without_mask/without_mask_${i}_64x64.jpg`));
    }
  }
}

function setup() {
  createCanvas(400, 400);

  let options = {
    inputs: [64, 64, 4],
    task: 'imageClassification',
    debug: true,
  };

  maskClassifier = ml5.neuralNetwork(options);

  // train the model on the current batch of images
  for (let i = 0; i < w_mask.length; i++) {
    maskClassifier.addData({ image: w_mask[i] }, { label: 'with mask' });
    maskClassifier.addData({ image: wo_mask[i] }, { label: 'without mask' });
  }

  maskClassifier.normalizeData();

  maskClassifier.train({ epochs: 10 }, () => {
    console.log('Finished training on batch ' + currentBatch);
    currentBatch++;

    if (currentBatch < totalBatches) {
      // clear the arrays for the next batch
      w_mask = [];
      wo_mask = [];

      // load the next batch of images
      loadBatch();

      // train the model on the next batch of images
      for (let i = 0; i < w_mask.length; i++) {
        maskClassifier.addData({ image: w_mask[i] }, { label: 'with mask' });
        maskClassifier.addData({ image: wo_mask[i] }, { label: 'without mask' });
      }

      maskClassifier.normalizeData();
      maskClassifier.train({ epochs: 10 }, finishedTraining);
    } else {
      // save the model after training on all batches
      maskClassifier.save();
      console.log('Finished training on all batches');
    }
  });
}

function finishedTraining() {
  console.log('Finished training on batch ' + currentBatch);
  currentBatch++;

  if (currentBatch < totalBatches) {
    // clear the arrays for the next batch
    w_mask = [];
    wo_mask = [];

    // load the next batch of images
    loadBatch();

    // train the model on the next batch of images
    for (let i = 0; i < w_mask.length; i++) {
      maskClassifier.addData({ image: w_mask[i] }, { label: 'with mask' });
      maskClassifier.addData({ image: wo_mask[i] }, { label: 'without mask' });
    }

    maskClassifier.normalizeData();
    maskClassifier.train({ epochs: 10 }, finishedTraining);
  } else {
    // save the model after training on all batches
    maskClassifier.save();
    console.log('Finished training on all batches');
  }
}