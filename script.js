document.addEventListener("DOMContentLoaded", function () {
  const images = document.querySelectorAll(".image");
  const placeholder = document.querySelector(".image-placeholder");

  images.forEach(function (image) {
    image.addEventListener("click", function () {
      const imageUrl = this.getAttribute("src");
      placeholder.style.backgroundImage = `url('${imageUrl}')`;
    });
  });
});
