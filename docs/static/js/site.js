document.addEventListener("DOMContentLoaded", () => {
  const title = document.title;
  if (title.includes("Project Website")) {
    document.body.style.background =
      "radial-gradient(circle at 16% 10%, rgba(70,194,166,0.2), transparent 35%)," +
      "radial-gradient(circle at 82% 22%, rgba(241,177,79,0.16), transparent 30%)," +
      "linear-gradient(155deg, #041019 0%, #0b1f2d 56%, #051017 100%)";
  } else if (title.includes("Platform Website")) {
    document.body.style.background =
      "radial-gradient(circle at 10% 12%, rgba(123,182,255,0.18), transparent 36%)," +
      "radial-gradient(circle at 88% 28%, rgba(70,194,166,0.16), transparent 34%)," +
      "linear-gradient(160deg, #07111a 0%, #0d2331 55%, #08131d 100%)";
  }
});
