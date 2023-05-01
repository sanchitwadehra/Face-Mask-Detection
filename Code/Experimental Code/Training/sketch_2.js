let w_mask=[];
let wo_mask=[];
let currentDataIndex = 849;
let numIterations = 3; // number of times to fine-tune the model

function preload(){
    for(let i=1;i<=currentDataIndex;i++){
        w_mask[i-1]=loadImage(`data/standardized/with_mask/with_mask_${i}_64x64.jpg`);
        wo_mask[i-1]=loadImage(`data/standardized/without_mask/without_mask_${i}_64x64.jpg`);
    }
}

function setup(){
    createCanvas(400,400);
    
    let options ={
        inputs: [64,64,4],
        task: 'imageClassification',
        debug: true,
    }
    maskClassifier=ml5.neuralNetwork(options);
    for(let i=0;i<w_mask.length;i++){
        maskClassifier.addData({image: w_mask[i]},{label: "with mask"});
        maskClassifier.addData({image: wo_mask[i]},{label: "without mask"});
    }
    maskClassifier.normalizeData();
    maskClassifier.train({epochs:200},finishedTraining);
}

function finishedTraining(){
    console.log("finished training");
    maskClassifier.save();
    for (let i = 0; i < numIterations; i++) {
      fineTuneModel();
    }
}

function fineTuneModel() {
  // add data from remaining images in w_mask and wo_mask folders
  for(let i = currentDataIndex; i < currentDataIndex + 849; i++) {
    let img1 = loadImage(`data/standardized/with_mask/with_mask_${i}_64x64.jpg`);
    let img2 = loadImage(`data/standardized/without_mask/without_mask_${i}_64x64.jpg`);
    maskClassifier.addData({image: img1},{label: "with mask"});
    maskClassifier.addData({image: img2},{label: "without mask"});
  }
  currentDataIndex += 849;
  
  // train the model with the new data
  maskClassifier.normalizeData();
  maskClassifier.train({epochs:200}, function() {
    console.log("fine-tuning completed");
    maskClassifier.save();
  });
}

/*
function draw(){
    background(0);
    image(w_mask[0],0,0,width,height);
}
*/
