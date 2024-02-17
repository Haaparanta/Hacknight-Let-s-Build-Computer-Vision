const container = document.querySelector(".container");

function handleImageUpload(e) {
  e.preventDefault();
  const data = new FormData(e.target);

  const file = data.get("file");
  const objectURL = URL.createObjectURL(file);

  fetch("/api/waldo", {
    method: "POST",
    body: data,
  }).then((response) => {
    clearContainer();
    response.json().then((body) => {
      const percentage = body["percentage"];

      const result = document.createElement("p");
      if (percentage === 1) {
        result.textContent = "Waldo found!";
      } else {
        result.textContent = "Waldo not found :/";
      }
      container.appendChild(result);

      const img = document.createElement("img");
      img.src = objectURL;
      container.appendChild(img);
    });
  });
}

function clearContainer() {
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
}
