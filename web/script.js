function parse_data(input){
    let output = []
    let current_state = {}
    listened_keys.forEach(e => current_state[e] = DEFAULT_VALUE)

    let index = 0
    for(let frame=0; frame<video.duration*FPS; frame++){
        while(index < input.length && input[index][0] == frame){
            if(listened_keys.includes(input[index][1])){
                current_state[input[index][1]] = input[index][2]
            }
            index++
        }
        output.push({...current_state}) // Make a copy
    }
    return output
}

function compute_accuracy(prediction, truth){
    let accuracy = {}
    listened_keys.forEach(e => accuracy[e] = 0)

    for(let frame=0; frame<video.duration*FPS; frame++){
        listened_keys.forEach(e => {
            if(prediction[frame][e] == truth[frame][e]){
                accuracy[e] += 1
            }
        })
    }
    return accuracy
}

function compute_transition_table(data){
    let state = []
    let output = []

    for(let i=0; i < listened_keys.length; i++){
        state[i] = []
        for(let j=0; j < listened_keys.length; j++){
            state[i][j] = 0
        }
    }

    let prev_index = listened_keys.indexOf(data[0][1])
    for(let i=0; i<data.length && data[i][0] < getCurrentFrame(); i++){
        if(listened_keys.includes(data[i][1]))
        {
            index = listened_keys.indexOf(data[i][1])
            state[index][prev_index] += 1
            output.push(JSON.parse(JSON.stringify(state)))
            prev_index = index
        }
    }
    return output
}

function show_transition_table(input, table){
    if(input.length == 0){
        return
    }

    table.innerHTML = ""
    let new_content = ""
    let index = input.length-1

    let sum = 0
    input[index].forEach(row => row.forEach(elem => sum += elem))

    for(let i=0; i < listened_keys.length; i++){

        if(i == 0){
            new_content += "<th>"
            for(let j=0; j<listened_keys.length; j++){
                new_content += "<td>" + listened_keys[j] + "</td>"
            }
            new_content += "</th>"
        }

        new_content += "<tr><td>" + listened_keys[i] + "</td>"
        for(let j=0; j < listened_keys.length; j++){
            value = (100*input[index][i][j]/sum).toFixed(1)
            new_content += "<td style='background-color:hsl(0, 100%, "+(100-value/2)+"%)'>" + value + "%</td>"
        }
        table.innerHTML += new_content + "</tr>"
        new_content = ""
    }
    
    
}