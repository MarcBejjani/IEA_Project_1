window.addEventListener('load', () => {
    const canvas = document.querySelector("#canvas");
    const button = document.querySelector('#button')
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    let painting = false;

    function startPosition(){
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

        ctx.lineTo(e.clientX, e.clientY);
        ctx.stroke();
        ctx.beginPath()
        ctx.moveTo(e.clientX, e.clientY)
    }
    function displayImage(){
        const img = canvas.toDataURL('image/png')
        document.getElementById("input").src= img;
        fileInput = document.getElementById("imageForm")
//        const myFile = new File([img], 'myFile.png', {
//        type: 'image/png',
//        lastModified: new Date(),
//        });
//        const dataTransfer = new DataTransfer();
//        dataTransfer.items.add(myFile);
//        fileInput.files = dataTransfer.files;

        var link = document.createElement('a');
        link.download = img;
        link.href = img;
        link.click();
        fileInput.files = link.href
    }
    //EventListeners
    canvas.addEventListener('mousedown', startPosition)
    canvas.addEventListener('mouseup', finishedPosition)
    canvas.addEventListener('mousemove', draw)
    button.addEventListener("click", displayImage);
})


window.addEventListener('resize', () => {
    //Resizing
    canvas.height = window.innerHeight;
    canvas.width = window.innerWidth;
})
