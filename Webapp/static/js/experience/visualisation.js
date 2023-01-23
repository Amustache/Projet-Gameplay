class Graph{
    constructor(container, data, title="", xTitle="", yTitle=""){
        this.container = container
        this.data = data
        this.title = title
        this.xTitle = xTitle
        this.yTitle = yTitle

        this.windowSize = 200

        this.axes = {x:"frame", y:"y", z:"key" , stroke:"key"}
        this.timeLine = [{x:0,y:0},{x:0,y:100}]
        this.render()
    }

    updateWindow(windowSize=0){
        this.windowSize = windowSize
        this.render()
    }

    updateTimeLine(currentFrame){
        this.timeLine = [{x:currentFrame,y:0},{x:currentFrame,y:100}]
        this.render()
    }

    render(){
        this.container.innerHTML = ""
        this.container.appendChild(Plot.plot({
            caption: this.title,
            width: this.container.clientWidth,
            y:{
                grid:true,
                label:this.yTitle
            },
            x:{
                label:this.xTitle
            },
            marks:[
                Plot.line(this.data, Plot.windowY({k:this.windowSize, anchor:"middle", strict:false}, this.axes )),
                Plot.line(this.timeLine,{x:'x',y:'y',stroke: "#FF0000"})
            ],
            color:{
                legend: true
            }
        }))
    }
}

class Markov{
    constructor(container, data, title=""){
        this.container = container
        this.data = data
        this.title = title
    }

    render(){
        this.container.innerHTML = ""
        this.container.appendChild(Plot.plot({
            caption: this.title,
            width: this.container.clientWidth,
            marks:[
                Plot.text(data,  {x:"",  y:"",  text:"", fontSize:10}),
                Plot.arrow(data, {x1:"", x2:"", y1:"",   y2:""})
            ],
            color:{
                legend: true
            }
        }))
    }
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