let maskClassifier;
let canvas;
let video;
let resultsDiv;

function setup(){


  const withMaskFolder = 'data/standardized/with_mask';
const withoutMaskFolder = 'data/standardized/without_mask';

// Load the images from the with mask folder
const withMaskImages = [];
for (let i = 0; i < 250; i++) {
  const randomIndex = Math.floor(Math.random() * 3725);
  const imagePath = `${withMaskFolder}/with_mask_${randomIndex}_64x64.jpg`;
  const image = loadImage(imagePath);
  withMaskImages.push(image);
}

// Load the images from the without mask folder
const withoutMaskImages = [];
for (let i = 0; i < 250; i++) {
  const randomIndex = Math.floor(Math.random() * 3828);
  const imagePath = `${withoutMaskFolder}/without_mask_${randomIndex}_64x64.jpg`;
  const image = loadImage(imagePath);
  withoutMaskImages.push(image);
}

  canvas=createCanvas(400,400);
  video=createCapture(VIDEO);
  video.size(64,64);
  
  let options ={
      inputs: [64,64,4],
      task: 'imageClassification',
  }
  maskClassifier=ml5.neuralNetwork(options);

  const modelDetails = {
      model: 'model/model.json',
      metadata: 'model/model_meta.json',
      weights: 'model/model.weights.bin'
    }
    maskClassifier.load(modelDetails,modelLoaded);
    resultsDiv=createDiv('loading model');
    
}

function modelLoaded(){
  console.log('model Ready!');
  classifyImage();
}

function classifyImage(){
  /*
  //let input=createGraphics(64,64);
  //input.copy(canvas,0,0,400,400,0,0,64,64);
  //image(input,0,0);
  maskClassifier.classify({image: video},gotResults);
  */
  const correct = 0;
  const total = withMaskImages.length + withoutMaskImages.length;
  
  for (const image of withMaskImages) {
    const prediction = maskClassifier.classify(image);
    if (prediction === image.split('_')[0]) {
      correct++;
    }
  }
  
  for (const image of withoutMaskImages) {
    const prediction = maskClassifier.classify(image);
    if (prediction === image.split('_')[0]) {
      correct++;
    }
  }
  
  // Calculate the accuracy
  const accuracy = correct / total * 100;
  console.log(`Accuracy: ${accuracy}%`);
}
/*
// Load the model
const classifier = ml5.imageClassifier('model.json');

// Classify the images
const correct = 0;
const total = withMaskImages.length + withoutMaskImages.length;

for (const image of withMaskImages) {
  const prediction = classifier.classify(image);
  if (prediction === image.split('_')[0]) {
    correct++;
  }
}

for (const image of withoutMaskImages) {
  const prediction = classifier.classify(image);
  if (prediction === image.split('_')[0]) {
    correct++;
  }
}

// Calculate the accuracy
const accuracy = correct / total * 100;
console.log(`Accuracy: ${accuracy}%`);
*/