function preprocess_element(element){
    let result = 0
    for(let i=0; i<listened_keys.length; i++){
        if(element[listened_keys[i]] == "DOWN"){
            result += Math.pow(2, i)
        }
    }
    return result.toString()
}

function unprocess_element(element){
    let result = []
    for(let i=0; i<element.length; i++){
        let state_num = parseInt(element[i])
        let current_key_state = []
        for(let j=listened_keys.length-1; j>=0; j--){
            if(state_num >= Math.pow(2, j)){
                current_key_state.push(listened_keys[j])
                state_num -= Math.pow(2, j)
            }
        }
        result.push(current_key_state.length == 0 ? ["None"] : current_key_state)
    }
    return result
}

function preprocess(data){
    let result = []
    let previous = undefined
    data.forEach(elem => {
        let preprocessed = preprocess_element(elem)
        if(preprocessed != previous && preprocessed != undefined){
            result.push(preprocessed)
            previous = preprocessed
        }
    })
    return result
}

function tokenize(elem_1, elem_2){
    return elem_1 +''+ elem_2
}

function BPE_iteration(data){
    let tokens = {}
    let best = {
        token: undefined,
        count: 0
    }

    for(let i=0; i<data.length-1; i++){

        let token = tokenize(data[i], data[i+1])

        if(tokens[token] == undefined){
            tokens[token] = 0
        }
        tokens[token]++
        if(tokens[token] > best.count){
            best.token = token
            best.count = tokens[token]
        }
    }

    let new_data = []
    for(let i=0; i<data.length; i++){
        if(i != data.length-1 && tokenize(data[i], data[i+1]) == best.token){
            new_data.push(best.token)
            i++
        }else{
            new_data.push(data[i])
        }
    }

    return [new_data, tokens]
}

function BPE(data, nb_iter=10){
    let trans_table

    for(let i=0; i<nb_iter; i++){
        console.log("Iteration "+i)

        let [new_data, tokens] = BPE_iteration(data)
        data = new_data

        let keys = []
        data.forEach(elem => {
            if(keys.indexOf(elem) == -1){
                keys.push(elem)
            }
        })

        trans_table = compute_transition_table_NLP(data, keys)

        let table = document.createElement("table")
        show_transition_table(trans_table, table, keys)

        let table_keys = document.createElement("table")
        let unproc_keys = unprocess_keys(keys)
        unproc_keys.forEach(key => table_keys.innerHTML += "<tr><td>"+JSON.stringify(key)+"</td></tr>")

        let div = document.createElement("div")
        div.appendChild(table)
        div.appendChild(table_keys)
        getById("parameters").appendChild(div)

        find_loop_in_ttable(convert_table_to_column_percent(trans_table.state))
    }

    return trans_table
}

function unprocess_keys(keys){
    let result = []
    keys.forEach(key => result.push(unprocess_element(key)))
    return result
}

function compute_transition_table_NLP(data, keys){
    let state = []
    for(let i=0; i < keys.length; i++){
        state[i] = []
        for(let j=0; j < keys.length; j++){
            state[i][j] = 0
        }
    }

    let prev_index = keys.indexOf(data[0])
    data.forEach(elem => {
        index = keys.indexOf(elem)
        state[index][prev_index] += 1
        prev_index = index
    })
    return {state:state}
}