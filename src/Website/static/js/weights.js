window.addEventListener('load', () => {
    const weights = document.getElementById("weights");
    const knn = document.getElementById('knn')
    const svm = document.getElementById('svm')
    const forest = document.getElementById('forest')
    const svmEnsemble = document.getElementById('svm-ensemble')
    const ensemble = document.getElementById('ensemble')
    const modelName = document.getElementById('modelName')

    function addWeights(){
        weights.style.display = 'block'
    }

    function removeWeights(){
        weights.style.display = 'none'
    }

    function showKNN(){
        modelName.textContent = 'KNN'

    }

    function showSVM(){
        modelName.textContent = 'SVM'
    }

    function showForest(){
        modelName.textContent = 'Random Forest'
    }

    function showSVMEnsemble(){
        modelName.textContent = 'Sequential Ensemble'
    }

    function showEnsemble(){
        modelName.textContent = 'Parallel Ensemble'
    }


    knn.addEventListener('click', removeWeights)
    svm.addEventListener('click', removeWeights)
    forest.addEventListener('click', removeWeights)
    svmEnsemble.addEventListener('click', removeWeights)
    ensemble.addEventListener('click', addWeights)

    knn.addEventListener('click', showKNN)
    svm.addEventListener('click', showSVM)
    forest.addEventListener('click', showForest)
    svmEnsemble.addEventListener('click', showSVMEnsemble)
    ensemble.addEventListener('click', showEnsemble)

})