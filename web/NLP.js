function preprocess_element(element){
    let result = 0
    for(let j=0; j<listened_keys.length; j++){
        if(element[listened_keys[j]] == "UP"){
            result += Math.pow(2, j)
        }
    }

    if(result != Math.pow(2, listened_keys.length)-1){
        return result
    }else{
        return undefined
    }
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
    console.log("Best : "+best.token)

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
    for(let i=0; i<nb_iter; i++){
        console.log("Iteration "+i)

        let [new_data, tokens] = BPE_iteration(data)
        console.log(tokens)
        console.log(new_data)
        data = new_data

        let keys = []
        data.forEach(elem => {
            if(keys.indexOf(elem) == -1){
                keys.push(elem)
            }
        })
        console.log(keys)
        let trans_table = compute_transition_table_NLP(data, keys)
        console.log(trans_table)
        let table = document.createElement("table")
        show_transition_table(trans_table, table, keys)
        getById("parameters").appendChild(table)
    }
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