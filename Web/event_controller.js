function changeImageNumber() {
    const numberOfImages = document.getElementById("imageNumberSelector").value; 
    const imageContainer = document.getElementById('imageContainer');  
    imageContainer.style.gridTemplateColumns = `repeat(${numberOfImages}, 1fr)`;  
    textContainer.textContent = `repeat(${numberOfImages}, 1fr)`; 
}


let saveIP = ""; 
function click_Image(imagePath) {
    idx = idx - 1; 
    tabIndex.set(imagePath, idx); 
    
    changeMappingTextBox(imagePath); 
    imagePaths = generateImagePaths(imagePath);
    
    const imageContainer = document.getElementById('imageContainer');  
    imageContainer.style.gridTemplateColumns = `repeat(${4}, 1fr)`;  
    showImages(imagePath); 
    // Show back button to allow user to go back to the initial state of showing all images
    if(document.getElementById('backButton').style.display !== 'inline') {
        saveIP = imagePath; 
        document.getElementById('backButton').style.display = 'inline';
    }
}


function click_Back() {
    imagePaths = copyList(originImagePaths); 
    
    changeImageNumber();
    showImages(saveIP); 
    // Hide back button after going back to all images
    document.getElementById('backButton').style.display = 'none'; 
}


function upload_CSV(csvContent) {
    const lines = csvContent.split('\n'); // Split the CSV into lines
    imagePaths = []; // Clear image paths, refresh 
    
    lines.forEach(line => {
        const [baseName, number] = line.split(','); // Split each line by comma
        if (baseName && number) {
            // Construct the image path
            const imagePath = createImagePath(baseName, number);
            imagePaths.push(imagePath); // Add to the array
        }
    });
    
    // Clean up and format image paths
    imagePaths = imagePaths.map(line => line.trim()).filter(path => path !== "");
    imagePaths = formatImagePaths(imagePaths);
    originImagePaths = copyList(imagePaths); 
    
    // Show images
    // changeImageNumber(); 
    showImages(""); 
    // Hide back button after going back to all images
    document.getElementById('backButton').style.display = 'none'; 
}


function click_Search() {
    // Get the value from the input field
    const searchInput = document.getElementById('searchInput').value;
    expression = searchInput; 
    // imagePaths = copyList(originImagePaths); 
    showImages("");
    // Hide back button after going back to all images
    document.getElementById('backButton').style.display = 'none'; 
}


function click_Clear() {
    // Get the value from the input field
    document.getElementById('searchInput').value = ""; 
    expression = null; 
    imagePaths = copyList(beforeImagePaths); 
    showImages("");
    // Non Hide back button after going back to all images
    document.getElementById('backButton').style.display = 'inline'; 
}


function click_Histogram() {
    const values = []; 
    const list = copyList(originImagePaths); 
    list.forEach((filePath, index) => {
        const baseFileName = filePath.substring(filePath.lastIndexOf(BACK) + 1, filePath.lastIndexOf('_'));
        values.push(baseFileName); 
    }); 
    showHistogram(values); 
    showPie(values); 
}


