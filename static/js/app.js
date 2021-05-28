var dataURI;
const vid = document.getElementById("myVideo");
function Redirect() {
  window.location = "http://localhost:5000/take_pic";
}
vid.onpause = function () {
  // alert("video paused");
  var canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 480;
  var ctx = canvas.getContext("2d");
  var vid = document.getElementById("myVideo");
  ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
  console.log("ok");
  //convert to desired file format
  dataURI = canvas.toDataURL("image/jpeg"); // can also use 'image/png'
  // myImage =document.getElementById("outputImage");
  // myImage.src=dataURI;
};

const img = document.getElementById("outputImage");
img.addEventListener("click", myFunction);
function myFunction() {
  var formData = new FormData();
  formData.append("string", dataURI);
  axios
    .post("http://localhost:5000/", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then((res) => {
      Redirect();
      console.log(res.data);
      const output = document.getElementById("display");
      for (let i = 0; i < 30; i++) {
        var x = document.createElement("IMG");
        x.setAttribute(
          "src",
          "static/data-v4/" +
            res.data["" + 31] +
            "/" +
            res.data["" + i] +
            ".jpg"
        );
        output.append(x);
      }
      // return (res.data);
    });
}
