let maskClassifier;
let canvas;
let video;
let resultsDiv;
let withMaskImages = [];
let withoutMaskImages = [];
const withMaskFolder = "data/standardized/with_mask";
const withoutMaskFolder = "data/standardized/without_mask";

function preload() {
  // Load the images from the with mask folder
  const selectedIndices = [];
  for (let i = 0; i < 500; i++) {
    let randomIndex;
    do {
      randomIndex = Math.floor(Math.random() * 3725);
    } while (selectedIndices.includes(randomIndex));

    selectedIndices.push(randomIndex);
    const imagePath = `${withMaskFolder}/with_mask_${randomIndex}_64x64.jpg`;
    const image = loadImage(imagePath);
    withMaskImages.push(image);
  }

  // Load the images from the without mask folder
  const selectedIndices2 = [];
  for (let i = 0; i < 500; i++) {
    let randomIndex;
    do {
      randomIndex = Math.floor(Math.random() * 3828);
    } while (selectedIndices2.includes(randomIndex));

    selectedIndices2.push(randomIndex);
    const imagePath = `${withoutMaskFolder}/without_mask_${randomIndex}_64x64.jpg`;
    const image = loadImage(imagePath);
    withoutMaskImages.push(image);
  }

  // Call the classifyImage function when the images are loaded
  Promise.all([...withMaskImages, ...withoutMaskImages]).then(() => {
    console.log("Images are loaded");
  });
}

function setup() {
  /*
  canvas = createCanvas(400, 400);
  video = createCapture(VIDEO);
  video.size(64, 64);
  */

  // Create a neural network classifier
  maskClassifier = ml5.neuralNetwork({
    inputs: [64, 64, 4],
    task: "imageClassification",
  });
  
/*
  resultsDiv = createDiv("Loading model...");
*/

  const modelDetails = {
    model: "model/model.json",
    metadata: "model/model_meta.json",
    weights: "model/model.weights.bin",
  };

  maskClassifier.load(modelDetails, () => {
    console.log("Model is loaded");
    classifyImage(withMaskImages, withoutMaskImages);
  });
}

function classifyImage(withMaskImages, withoutMaskImages) {
  const total = withMaskImages.length + withoutMaskImages.length;
  let correct = 0;
  let numClassified = 0;

  for (const image of withMaskImages) {
    const imageObj = { image };
    maskClassifier.classify(imageObj, (error, result) => {
      if (error) {
        console.error(error);
      } else {
        if (result[0].label === "with mask") {
          correct++;
        }
        numClassified++;
        if (numClassified === total) {
          const accuracy = (correct / total) * 100;
          console.log(`Accuracy: ${accuracy}%`);
        }
      }
    });
  }

  for (const image of withoutMaskImages) {
    const imageObj = { image };
    maskClassifier.classify(imageObj, (error, result) => {
      if (error) {
        console.error(error);
      } else {
        if (result[0].label === "without mask") {
          correct++;
        }
        numClassified++;
        if (numClassified === total) {
          const accuracy = (correct / total) * 100;
          console.log(`Accuracy: ${accuracy}%`);
        }
      }
    });
  }
}
