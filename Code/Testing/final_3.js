

const withMaskFolder = 'data/standardized/with_mask';
const withoutMaskFolder = 'data/standardized/without_mask';

// Load the images from the with mask folder
const withMaskImages = [];
for (let i = 0; i < 250; i++) {
  const randomIndex = Math.floor(Math.random() * 3725);
  withMaskImages.push(`${withMaskFolder}/with_mask_${randomIndex}_64x64.jpg`);
}

// Load the images from the without mask folder
const withoutMaskImages = [];
for (let i = 0; i < 250; i++) {
  const randomIndex = Math.floor(Math.random() * 3828);
  withoutMaskImages.push(`${withoutMaskFolder}/without_mask_${randomIndex}_64x64.jpg`);
}

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
