window.addEventListener('load', () => {
    const canvas = document.querySelector("#canvas");
    const button = document.querySelector('#button')
    const drawButton = document.querySelector('#drawButton')
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    var rect = canvas.getBoundingClientRect()
    var heightOffset = rect.top
    var widthOffset = rect.left

    var painting = false;

    function startPosition(e){
        painting = true;
        draw(e)
    }
    function finishedPosition(){
        painting = false;
        ctx.beginPath()
    }
    function draw(e){
        if(!painting) return;
        ctx.lineWidth = 5;
        ctx.lineCap = "round";

        ctx.lineTo(e.clientX-widthOffset, e.clientY-heightOffset);
        ctx.stroke();
        ctx.beginPath()
        ctx.moveTo(e.clientX-widthOffset, e.clientY-heightOffset)
    }
    function drawImage(){
        var link = document.createElement('a');
        link.href = '/draw';
        link.click();
    }
    function doFunction()
    {
        const img = canvas.toDataURL('image/png')

        $.post('/draw', {
            js_data: img,
        })

        var link = document.createElement('a');
        link.href = '/';
        link.click();
    };
    //EventListeners
    canvas.addEventListener('mousedown', startPosition)
    canvas.addEventListener('mouseup', finishedPosition)
    canvas.addEventListener('mousemove', draw)
    button.addEventListener("click", doFunction);
    drawButton.addEventListener("click", drawImage)
})



