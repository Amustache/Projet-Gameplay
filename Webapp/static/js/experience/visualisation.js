function addScatterGraph(container, datasets, xTitle="", yTitle="", height=GRAPH_DEFAULT_HEIGHT){
    /*let canvas = document.createElement("canvas")
    canvas.height = height
    canvas.width = "100%"

    let container_div = document.createElement("div")
    container_div.style.height = height+"px"
    container_div.appendChild(canvas)
    
    charts.push(new Chart(canvas, {
        type: 'scatter',
        data:{ datasets: datasets },
        options: {...CHART_OPTIONS, scales:{
            x : {title : {
                display : true,
                text : xTitle
            }},
            y : {title : {
                display : true,
                text : yTitle
            }}
        }}
    }))
    container.appendChild(container_div)*/
    console.log(datasets)
    let axes = {x:"frame", y:"y", z:"key" , stroke:"key"}
    container.appendChild(Plot.plot({
        y:{
            grid:true,
            label:"Accuracy [%]"
        },
        marks:[
            //Plot.line(datasets, axes),
            Plot.line(
                datasets, 
                Plot.windowY({k:200, anchor:"middle", strict:false}, axes )
            ),
            Plot.line([{x:5000,y:0},{x:5000,y:100}],{x:'x',y:'y',stroke: "#FF0000"})
        ]
    }))
}

function getTTableRowSums(ttable, keys){
    let sum = Array(keys.length).fill(0)
    ttable.state.forEach(row => {
        for(let i=0; i<row.length; i++){
            sum[i] += row[i]
        }
    })
    return sum
}

function printTransitionTable(ttable, tableHtml, keys){
    tableHtml.innerHTML = ""
    let newContent = ""
    let rowSums = getTTableRowSums(ttable, keys)

    newContent += "<td>↙</td>"
    keys.forEach(key => newContent += "<td>"+key+"</td>")

    for(let i=0; i<keys.length; i++){    
        newContent += "<tr><td>" + keys[i] + "</td>"
        for(let j=0; j<keys.length; j++){
            let value = rowSums[j] == 0 ? 0 : 100*ttable.state[i][j]/rowSums[j]
            let textColor = value==0 ? "text-black-50" : "text-black"
            let backgroundColor = "hsl(0, 100%, "+(100-value/2)+"%)"
            newContent += "<td class='"+textColor+"' style='background-color:"+backgroundColor+"'>" + value.toFixed(1) + "%</td>"
        }
        tableHtml.innerHTML += newContent + "</tr>"
        newContent = ""
    }
}