function find_loop_in_ttable(ttable){
    let processed = convert_table_to_column_percent(ttable)
    let result = []
    for(let i=0; i<processed.length; i++){
        let loops = find_loop_in_table_entry(processed, i)
        let best_loop = {probability:0}
        loops.forEach(loop => {
            if(loop.probability > best_loop.probability){
                best_loop = loop
            }
        })
        result.push(best_loop)
    }
    console.log(result)

    let best = {probability:0}
    result.forEach(elem => {
        if(elem.probability > best.probability){
            best = elem
        }
    })

    console.log(best)
    return best
}

// find_loop_in_table_entry(convert_table_to_column_percent(BPE(preprocess(parsed_prediction), 1).state),1) 
function find_loop_in_table_entry(ttable, entry_index){
    let graph = {
        entry_index : entry_index,
        visited_indices: [],
        candidates: [{
            path:[entry_index],
            probability: 1,
            visited: false
        }],
        best_candidate_index: 0
    }

    let loops = []
    while(graph.best_candidate_index != -1){

        let best_candidate = graph.candidates[graph.best_candidate_index]
        let visited_index = best_candidate.path[best_candidate.path.length-1]
        graph.visited_indices.push(visited_index)
        get_next_candidates(ttable, best_candidate).forEach(candidate => {
            if(candidate.path[candidate.path.length-1] == entry_index){
                loops.push(candidate)
            }else if(graph.visited_indices.indexOf(candidate.path[candidate.path.length-1]) == -1){
                graph.candidates.push(candidate) 
            }
        })
        best_candidate.visited = true

        graph.best_candidate_index = -1
        for(let i=0; i<graph.candidates.length; i++){
            if(!graph.candidates[i].visited && graph.visited_indices.indexOf(graph.candidates[i].path[graph.candidates[i].path.length-1]) == -1){
                if(graph.best_candidate_index == -1 || graph.candidates[i].probability > graph.candidates[graph.best_candidate_index].probability){
                    graph.best_candidate_index = i
                }
            }
        }
    }

    //console.log(graph)
    
    return loops
}

function get_next_candidates(ttable, candidate){
    let result = []
    let entry_index = candidate.path[candidate.path.length-1]
    for(let i=0; i<ttable.length; i++){
        if(ttable[i][entry_index] != 0){
            result.push({
                path: candidate.path.concat([i]),
                probability: candidate.probability * ttable[i][entry_index],
                visited: false
            })
        }
    }
    return result
}

function convert_table_to_column_percent(ttable){
    let result = []

    let sums = Array(ttable.length).fill(0)
    ttable.forEach(row => {
        for(let i=0; i<row.length; i++){
            sums[i] += row[i]
        }
    })

    for(let i=0; i<ttable.length; i++){
        result.push([])
        for(let j=0; j<ttable[i].length; j++){
            result[i].push(ttable[i][j]/sums[j])
        }
    }
    return result
}