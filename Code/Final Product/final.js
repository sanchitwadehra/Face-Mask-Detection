let maskClassifier;
let canvas;
let video;
let resultsDiv;


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

function classifyImage(){
    //let input=createGraphics(64,64);
    //input.copy(canvas,0,0,400,400,0,0,64,64);
    //image(input,0,0);
    maskClassifier.classify({image: video},gotResults);
}

function gotResults(err,results){
    if(err){
        console.error(err);
    }
    console.log(results);
    let label=results[0].label;
    let confidence=nf(100*results[0].confidence,2,0);
    resultsDiv.html(`${label} ${confidence}%`);
    classifyImage();
}

function modelLoaded(){
    console.log('model Ready!');
    classifyImage();
}

function draw(){
    image(video,0,0,350,350);
    
      //  background(0);
      //  image(w_mask[27],0,0,width,height);
       
}
/*
function draw(){
    background(0);
    image(w_mask[0],0,0,width,height);
}
*/