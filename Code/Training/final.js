let w_mask=[];
let wo_mask=[];

function preload(){
    for(let i=1;i<500;i++){
        w_mask[i-1]=loadImage(`data/standardized/with_mask/with_mask_${i}_64x64.jpg`);
    }
    for(let i=1;i<500;i++){
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
}

/*
function draw(){
    background(0);
    image(w_mask[0],0,0,width,height);
}
*/