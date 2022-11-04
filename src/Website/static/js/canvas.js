window.addEventListener('load', () => {
    const canvas = document.querySelector("#canvas");
    const button = document.querySelector('#button')
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
    function displayImage(){
        const img = canvas.toDataURL('image/png')
        var link = document.createElement('a');
        link.download = img;
        link.href = img;
        link.click();
    }
    function doFunction()
    {
        const img = canvas.toDataURL('image/png')
        data = {'url': img}
        $.ajax({
            type: 'POST',
            contentType: 'application/json',
            url: '/',
            dataType : 'json',
            data : JSON.stringify(data),
            success : function(result) {
              jQuery("#clash").html(result);
            },error : function(result){
               console.log(result);
            }
        });
    };
    //EventListeners
    canvas.addEventListener('mousedown', startPosition)
    canvas.addEventListener('mouseup', finishedPosition)
    canvas.addEventListener('mousemove', draw)
    button.addEventListener("click", doFunction);
})



