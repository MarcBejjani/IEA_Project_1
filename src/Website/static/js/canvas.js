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



//
//window.addEventListener('load', function () {
//// get the canvas element and its context
//var canvas = document.getElementById('canvas');
//var context = canvas.getContext('2d');
//
//// create a drawer which tracks touch movements
//var drawer = {
//    isDrawing: false,
//    touchstart: function (coors) {
//        context.beginPath();
//        context.moveTo(coors.x, coors.y);
//        this.isDrawing = true;
//    },
//    touchmove: function (coors) {
//        if (this.isDrawing) {
//            context.lineTo(coors.x, coors.y);
//            context.stroke();
//        }
//    },
//    touchend: function (coors) {
//        if (this.isDrawing) {
//            this.touchmove(coors);
//            this.isDrawing = false;
//        }
//    }
//};
//// create a function to pass touch events and coordinates to drawer
//function draw(event) {
//    var type = null;
//    // map mouse events to touch events
//    switch(event.type){
//        case "mousedown":
//                event.touches = [];
//                event.touches[0] = {
//                    pageX: event.pageX,
//                    pageY: event.pageY
//                };
//                type = "touchstart";
//        break;
//        case "mousemove":
//                event.touches = [];
//                event.touches[0] = {
//                    pageX: event.pageX,
//                    pageY: event.pageY
//                };
//                type = "touchmove";
//        break;
//        case "mouseup":
//                event.touches = [];
//                event.touches[0] = {
//                    pageX: event.pageX,
//                    pageY: event.pageY
//                };
//                type = "touchend";
//        break;
//    }
//
//    // touchend clear the touches[0], so we need to use changedTouches[0]
//    var coors;
//    if(event.type === "touchend") {
//        coors = {
//            x: event.changedTouches[0].pageX,
//            y: event.changedTouches[0].pageY
//        };
//    }
//    else {
//        // get the touch coordinates
//        coors = {
//            x: event.touches[0].pageX,
//            y: event.touches[0].pageY
//        };
//    }
//    type = type || event.type
//    // pass the coordinates to the appropriate handler
//    drawer[type](coors);
//}
//
//// detect touch capabilities
//var touchAvailable = ('createTouch' in document) || ('ontouchstart' in window);
//
//// attach the touchstart, touchmove, touchend event listeners.
//if(touchAvailable){
//    canvas.addEventListener('touchstart', draw, false);
//    canvas.addEventListener('touchmove', draw, false);
//    canvas.addEventListener('touchend', draw, false);
//}
//// attach the mousedown, mousemove, mouseup event listeners.
//else {
//    canvas.addEventListener('mousedown', draw, false);
//    canvas.addEventListener('mousemove', draw, false);
//    canvas.addEventListener('mouseup', draw, false);
//}
//
//// prevent elastic scrolling
//document.body.addEventListener('touchmove', function (event) {
//    event.preventDefault();
//}, false); // end body.onTouchMove
//
//}, false); // end window.onLoad



