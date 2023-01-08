function add_new_scatter_graph(container, datasets, height){
    let canvas = document.createElement("canvas")
    canvas.height = height
    canvas.width = "1400"
    let container_div = document.createElement("div")
    container_div.style.width = "90%"
    container_div.style.height = height+"px"
    container_div.appendChild(canvas)
    charts.push(new Chart(canvas, {
        type: 'scatter',
        data:{ datasets: datasets },
        options: CHART_OPTIONS
    }))
    container.appendChild(container_div)
}

function get_ttable_row_sums(ttable, keys){
    let sum = Array(keys.length).fill(0)
    ttable.state.forEach(row => {
        for(let i=0; i<row.length; i++){
            sum[i] += row[i]
        }
    })
    return sum
}

function print_ttable(ttable, table_html, keys){
    table_html.innerHTML = ""
    let new_content = ""
    let row_sums = get_ttable_row_sums(ttable, keys)

    new_content += "<th>"
    keys.forEach(key => new_content += "<td>"+key+"</td>")
    new_content += "</th>"

    for(let i=0; i < keys.length; i++){    
        new_content += "<tr><td>" + keys[i] + "</td>"
        for(let j=0; j < keys.length; j++){
            let value = (100*ttable.state[i][j]/row_sums[j]).toFixed(1)
            if(value != 0){
                new_content += "<td style='background-color:hsl(0, 100%, "+(100-value/2)+"%)'>" + value + "%</td>"
            }else{
                new_content += "<td></td>"
            }
        }
        table_html.innerHTML += new_content + "</tr>"
        new_content = ""
    }
}