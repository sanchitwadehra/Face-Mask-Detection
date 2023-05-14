let maskClassifier;
let canvas;
let video;
let resultsDiv;
const withMaskFolder = "data/standardized/with_mask";
const withoutMaskFolder = "data/standardized/without_mask";

function preload() {
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
  
    // Call the classifyImage function when the images are loaded
    Promise.all([...withMaskImages, ...withoutMaskImages]).then(() => {
      classifyImage(withMaskImages, withoutMaskImages);
    });
  }

function setup() {
  canvas = createCanvas(400, 400);
  video = createCapture(VIDEO);
  video.size(64, 64);

  // Create a neural network classifier
  maskClassifier = ml5.neuralNetwork({
    inputs: [64, 64, 4],
    task: "imageClassification",
  });

  resultsDiv = createDiv("Loading model...");

  const modelDetails = {
    model: "model/model.json",
    metadata: "model/model_meta.json",
    weights: "model/model.weights.bin",
  };

  maskClassifier.load(modelDetails, () => {
    console.log("Model is loaded");
  });
}

function classifyImage(withMaskImages, withoutMaskImages) {
    const total = withMaskImages.length + withoutMaskImages.length;
    let correct = 0;
  
    for (const image of withMaskImages) {
      const imageObj = { image };
      maskClassifier.classify(imageObj, (error, result) => {
        if (error) {
          console.error(error);
        } else {
          if (result[0].label === "with_mask") {
            correct++;
          }
          const accuracy = (correct / total) * 100;
          console.log(`Accuracy: ${accuracy}%`);
        }
      });
    }
  
    for (const image of withoutMaskImages) {
      const imageObj = { image };
      maskClassifier.classify(imageObj, (error, result) => {
        if (error) {
          console.error(error);
        } else {
          if (result[0].label === "without_mask") {
            correct++;
          }
          const accuracy = (correct / total) * 100;
          console.log(`Accuracy: ${accuracy}%`);
        }
      });
    }
  }
  
