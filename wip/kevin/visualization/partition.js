import { prediction_data } from "../data/visu/videos-smb-yt/smb4.js";

const VIDEO = document.getElementById("video")
console.log(VIDEO.duration)
const FPS = 30

let radius = 20

let margin = {top: 10, right: 40, bottom: 30, left: 30},
    width = 1200 - margin.left - margin.right,
    height = 800 - margin.top - margin.bottom;

let actions = ['Left', 'Jump', 'Right']

let keys = getKeysFromArray(getCleanArray(prediction_data))
let rectPos = createRectFromCleanArray(getCleanArray(prediction_data), keys);


let sVg = d3.select('#area')
    .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")

// X scale and Axis
var startTime = new Date(0,0,0);
var endTime = new Date(startTime.getTime() + (31.3 * 60 * 1000));

var x = d3.scaleTime()
    .domain([startTime, endTime])
    .range([0, width])
    .nice();   

let xAxis = sVg.append("g")
    .attr("transform", "translate(" + 2*radius + "," + height + ")")
    .call(d3.axisBottom(x));


// Y scale and Axis
let y = d3.scaleBand()
    .domain(actions)        
    .range([0, height])
    .padding(1);       

let yAxis = sVg.append('g')
    .attr("transform", "translate(" + 2*radius + ",0)")
    .call(d3.axisLeft(y))
    .selectAll("text").remove();




let right_arrow = sVg.append('image')
    .attr('xlink:href', './arrows/right-arrow.svg')
    .attr('width', 60)
    .attr('height', 30)
    .attr('x', x(0) - 20)
    .attr('y', y('Right') - 15);

let left_arrow = sVg.append('image')
    .attr('xlink:href', './arrows/left-arrow.svg')
    .attr('width', 60)
    .attr('height', 30)
    .attr('x', x(0) - 20)
    .attr('y', y('Left') - 15);

let up_arrow = sVg.append('image')
    .attr('xlink:href', './arrows/up-arrow.svg')
    .attr('width', 60)
    .attr('height', 30)
    .attr('x', x(0) - 20)
    .attr('y', y('Jump') - 15)



var clip = sVg.append("defs").append("sVg:clipPath")
    .attr("id", "clip")
    .append("sVg:rect")
    .attr("width", width )
    .attr("height", height )
    .attr("x", margin.right)
    .attr("y", 0);

var rects = sVg.append('g')
    .attr("clip-path", "url(#clip)")


rects.selectAll("rect")
    .data(rectPos)
    .enter()
    .append("rect")
        .attr("width", function(d){return x(d.width)})
        .attr("x", function(d){return x(d.x)})
        .attr("y", function(d){return y(d.y) - d.height/2})
        .attr("height", function(d){return d.height})
        .attr("transform", "translate(" + 2*radius + ", 0)")
        .attr("fill", function(d){return d.fill})



VIDEO.addEventListener("timeupdate", () => {
    rects.attr("x", d => x(VIDEO.currentTime))
})


function createRectFromCleanArray(clean_array, keys){

    let positions = []

    for(let i = 0, end ; i < keys.length ; i++){
        switch(keys[i]){
            case 'R':
                end = keys.slice(i).indexOf("r") + i;
                positions.push({x:(clean_array[i][0]/30), 
                                y:'Right',
                                width: (clean_array[end][0]/30) - (clean_array[i][0]/30),
                                height:30,
                                fill:'#d62728'})
                break;
            case 'J':
                end = keys.slice(i).indexOf("j") + i;
                positions.push({x:(clean_array[i][0]/30), 
                                y:'Jump',
                                width: (clean_array[end][0]/30) - (clean_array[i][0]/30),
                                height:30,
                                fill:'#2ca02c'})
                break;
            case 'L': 
                end = keys.slice(i).indexOf("l") + i;
                positions.push({x:(clean_array[i][0]/30), 
                                y:'Left',
                                width: (clean_array[end][0]/30) - (clean_array[i][0]/30),
                                height:30,
                                fill: '#1f77b4'})
                break;
            default:
                break;
        }
    }

    return positions
}

function getCleanArray(array){
    let array_clean = array.map(row => {
        switch(row[1]){
            case 'x':
                return (row[2] == 'DOWN' ? [row[0], 'J'] : [row[0], 'j'])
            case 'Key.right':
                return (row[2] == 'DOWN' ? [row[0], 'R'] : [row[0], 'r'])
            case 'Key.left':
                return (row[2] == 'DOWN' ? [row[0], 'L'] : [row[0], 'l'])
        }
    })

    return array_clean

}

function getKeysFromArray(array){
    let keys = ''
    array.forEach((elem, index) => {
        keys = keys + elem[1]
    });

    return keys
}
