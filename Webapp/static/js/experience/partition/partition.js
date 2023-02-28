function compute_partition(prediction_data) {
    const FPS = 25
    let window_size = 5

    let margin = {top: 10, right: 40, bottom: 30, left: 30},
        width = 800 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom,
        radius = 20;

    let actions = ['Left', 'Jump', 'Right']
    let keys = getKeysFromArray(getCleanArray(prediction_data))
    let rectPos = createRectFromCleanArray(getCleanArray(prediction_data), keys);


    let x = d3.scaleLinear()
                .domain([0,window_size])
                .range([0, width]).nice()

    let y = d3.scaleBand()
                .domain(actions)        
                .range([0, height])
                .padding(1);   
    

    let xAxis = d3.axisBottom(x)
                    .tickFormat(formatTimeFromSeconds())
                    .tickValues(d3.range(0, VIDEO.duration, 1));

    let yAxis = d3.axisLeft(y)

    let svg = createAreaAndChart()

    let clip = svg.append("defs")
                .append("svg:clipPath")
                    .attr("id", "clip")
                .append("svg:rect")
                    .attr("width", width )
                    .attr("height", height )
                    .attr("x", margin.right)
                    .attr("y", 0);

    let rects = svg.append('g')
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

    VIDEO.addEventListener("timeupdate", updateAxis);


    
    function createAreaAndChart(){

        let svg = d3.select('#area')
                    .append("svg")
                        .attr("width", width + margin.left + margin.right)
                        .attr("height", height + margin.top + margin.bottom)
                    .append("g")
                        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
                            

        let xGroup = svg.append("g")
            .attr("class", "x-axis")
            .attr("transform", "translate(" + 2*radius + "," + height + ")")
            .call(xAxis)   

        let yGroup = svg.append('g')
            .attr("class", "y-axis")
            .attr("transform", "translate(" + 2*radius + ",0)")
            .call(yAxis)
            .selectAll("text").remove();                


        for (let dir of actions){
            svg.append('image')
                .attr('xlink:href', './static/js/experience/partition/' + dir.toLowerCase() + '-arrow.svg')
                .attr('width', 60)
                .attr('height', 30)
                .attr('x', x(0) - 20)
                .attr('y', y(dir) - 15);
        }

        return svg
    }

    function createRectFromCleanArray(clean_array, keys){

        let positions = []

        for(let i = 0, end ; i < keys.length ; i++){
            switch(keys[i]){
                case 'R':
                    end = keys.slice(i).indexOf("r") + i;
                    positions.push({x:(clean_array[i][0]/FPS), 
                                    y:'Right',
                                    width: (clean_array[end][0]/FPS) - (clean_array[i][0]/FPS),
                                    height:30,
                                    fill:'#d62728'})
                    break;
                case 'J':
                    end = keys.slice(i).indexOf("j") + i;
                    positions.push({x:(clean_array[i][0]/FPS), 
                                    y:'Jump',
                                    width: (clean_array[end][0]/FPS) - (clean_array[i][0]/FPS),
                                    height:30,
                                    fill:'#2ca02c'})
                    break;
                case 'L': 
                    end = keys.slice(i).indexOf("l") + i;
                    positions.push({x:(clean_array[i][0]/FPS), 
                                    y:'Left',
                                    width: (clean_array[end][0]/FPS) - (clean_array[i][0]/FPS),
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
            switch(row["KEY"]){
                case 'x':
                    return (row["STATUS"] == 'DOWN' ? [row["FRAME"], 'J'] : [row["FRAME"], 'j'])
                case 'Key.right':
                    return (row["STATUS"] == 'DOWN' ? [row["FRAME"], 'R'] : [row["FRAME"], 'r'])
                case 'Key.left':
                    return (row["STATUS"] == 'DOWN' ? [row["FRAME"], 'L'] : [row["FRAME"], 'l'])
            }
        })
        return array_clean

    }

    function getKeysFromArray(array){
        let keys = ''
        array.forEach(elem => {
            keys = keys + elem[1]
        });
        return keys
    }

    function formatTimeFromSeconds(){
        return function(s) {
            let hours = Math.floor(s / 3600),
                minutes = Math.floor((s - (hours * 3600)) / 60),
                seconds = s - (minutes * 60);
            let output = ''

            if (seconds)
                output = seconds + 's';

            if (minutes || (!hours && !minutes)) {
                output = minutes + 'm ' + output;
            }
            if (hours) {
                output = hours + 'h ' + output;
            }
            
            return output;
        };
    }
    
    function updateAxis() {
        let currentTime = VIDEO.currentTime;
        x.domain([currentTime, currentTime + window_size]).range([0, width]);
    
        d3.select(".x-axis")
        .transition()
        .ease(d3.easeLinear)
        .call(xAxis);

        rects.selectAll("rect")
                .transition()
                .ease(d3.easeLinear)
                .attr("x", function(d){return x(d.x)})
                
        svg.append("g")
        .append('rect')
                .attr('fill', 'white')
                .attr('width', 65)
                .attr('height', margin.bottom)
                .attr('x', -margin.left)
                .attr('y', height)

    }
};
