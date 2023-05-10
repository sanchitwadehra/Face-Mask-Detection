let maskClassifier;
let w_mask = [];
let wo_mask = [];
let num_images_to_load = 450;
let progressBar;

function preload() {
  // Load 450 images with mask
  for (let i = 0; i < num_images_to_load; i++) {
    let img_index = Math.floor(Math.random() * 3725) + 1;
    w_mask[i] = loadImage(`data/standardized/with_mask/with_mask_${img_index}_64x64.jpg`);
  }
  
  // Load 450 images without mask
  for (let i = 0; i < num_images_to_load; i++) {
    let img_index = Math.floor(Math.random() * 3828) + 1;
    wo_mask[i] = loadImage(`data/standardized/without_mask/without_mask_${img_index}_64x64.jpg`);
  }
}

function setup() {
  createCanvas(400, 400);
  progressBar = createProgressBar(width/2, height/2, 200, 20);

  let options = {
    inputs: [64, 64, 4],
    task: 'imageClassification',
    debug: true,
  }

  // Initialize the model first
  maskClassifier = ml5.neuralNetwork(options);
  maskClassifier.load({
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
  }, function () {
    console.log('Model loaded');
    // Call your testing function here
    testModel();
  });
}

function testModel() {
  let correct = 0;
  let total = 0;

  for (let i = 0; i < w_mask.length; i++) {
    maskClassifier.classify({ image: w_mask[i] }, function (error, result) {
      if (error) {
        console.error(error);
      } else {
        if (result.label === "with mask") {
          correct++;
        }
        total++;
        progressBar.updateProgress((correct + (num_images_to_load - total))/ (2 * num_images_to_load));
        if (total === w_mask.length + wo_mask.length) {
          for (let j = 0; j < wo_mask.length; j++) {
            maskClassifier.classify({ image: wo_mask[j] }, function (error, result) {
              if (error) {
                console.error(error);
              } else {
                if (result.label === "without mask") {
                  correct++;
                }
                total++;
                progressBar.updateProgress((correct + (num_images_to_load - total))/ (2 * num_images_to_load));
                if (total === w_mask.length + wo_mask.length) {
                  let accuracy = (correct / total) * 100;
                  console.log(`Accuracy: ${accuracy.toFixed(2)}%`);
                }
              }
            });
          }
        }
      }
    });
  }
}

function createProgressBar(x, y, w, h) {
  let bar = {
    x: x,
    y: y,
    w: w,
    h: h,
    progress: 0,
    updateProgress: function(percentage) {
      this.progress = percentage;
      this.draw();
    },
    draw: function() {
      stroke(0);
      fill(255);
      rect(this.x - this.w/2, this.y - this.h/2, this.w, this.h);
      fill(0, 255, 0);
      rect(this.x - this.w/2, this.y - this.h/2, this.w * this.progress, this.h);
    }
  }
  return bar;
}
