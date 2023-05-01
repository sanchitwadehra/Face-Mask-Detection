let maskClassifier;
let canvas;
let video;
let w_mask=[];
let wo_mask=[];
let resultsDiv;
let testImages;
let testLabels;

function preload(){
    for(let i=3277;i<3524;i++){
        w_mask[i-3277]=loadImage(`data/standardized/with_mask/with_mask_${i}_64x64.jpg`);
    }
    for(let i=3017;i<3212;i++){
        wo_mask[i-3017]=loadImage(`data/standardized/without_mask/without_mask_${i}_64x64.jpg`);
    }
}
function setup(){
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

function mousePressed(){
    maskClassifier.classify({image: w_mask[27]},gotResults);
}

function gotResults(err,results){
    if(err){
        console.error(err);
    }
    console.log(results);
    let label=results[0].label;
    let confidence=nf(100*results[0].confidence,2,0);
    resultsDiv.html(`${label} ${confidence}%`);
}

function modelLoaded(){
    console.log('model Ready!');
}

function draw(){
    image(video,0,0,width,height);
    
        background(0);
        image(w_mask[27],0,0,width,height);
       
}
/*
function draw(){
    background(0);
    image(w_mask[0],0,0,width,height);
}
*/