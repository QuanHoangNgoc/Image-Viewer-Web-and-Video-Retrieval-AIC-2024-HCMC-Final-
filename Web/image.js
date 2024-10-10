/* define */ 
let imagePaths = []; /* display-need images (after filter) */ 
let beforeImagePaths = []; /* show-need images (before filter) */ 
let originImagePaths = []; /* file-based images */ 
let tabIndex = new Map(); /* order? of clicked image */ 
let idx = 0; 
let expression = ""; /* filter expression */ 
let DIFF = 10; 


function formatImagePaths(list) {
    const result = [];
    list.forEach((imagePath, index) => {
        newFilePath = imagePath.replace(/\//g, BACK);
        result.push(newFilePath);
    });
    return result; 
}

function copyList(list) {
    const result = []; 
    list.forEach(x => {
        result.push(x); 
    }); 
    return result; 
}

function showImages(centerPath) {
    const imageContainer = document.getElementById('imageContainer');
    imageContainer.innerHTML = ''; // Clear any previous images
    
    let center = 0; 
    beforeImagePaths = copyList(imagePaths); 
    if(expression) {
        imagePaths = getExpressionImages(expression); 
    }
    imagePaths.forEach((imagePath, index) => {
        // Load the image from database 
        const img = document.createElement('img');
        img.src = imagePath;
        img.alt = `Image ${index + 1}`;
        img.tabIndex = -1; // Ensure tab to back, not can use tab with image 
        
        // Extract folder and file name from the image path
        const folderPath = imagePath.substring(0, imagePath.lastIndexOf(BACK));
        const fileName = imagePath.substring(imagePath.lastIndexOf(BACK) + 1);
        // Add title attribute to display the folder and file info on hover
        img.title = `Folder: ${folderPath}\nFile: ${fileName}`;
        
        // Highlight the image IF was clicked
        if(tabIndex.has(imagePath)) {
            img.classList.add('highlighted');
        }
        // Find center
        if(imagePath === centerPath) {
            center = index; 
        }
        
        // Add click event
        img.addEventListener('click', () => click_Image(imagePath));  
        // Display the image 
        imageContainer.appendChild(img);
    });
    // Move to the focus image
    imageContainer.children[center].focus(); 
    // imageContainer.children[center].tabIndex = 1; 
}

function generateImagePaths(filePath) {
    // Extract the folder path and base file name
    const folderPath = filePath.substring(0, filePath.lastIndexOf(BACK) + 1);
    const baseFileName = filePath.substring(filePath.lastIndexOf(BACK) + 1, filePath.lastIndexOf('_'));
    const extension = filePath.substring(filePath.lastIndexOf('.'));
    const numFileName = filePath.substring(filePath.lastIndexOf('_') + 1, filePath.lastIndexOf('.')); 
    
    let num = parseInt(numFileName, 10);
    let startNumber = Math.max(num - DIFF + 1, 1);
    let endNumber = num + DIFF; 
    const imagePaths = [];
    
    // Generate image paths from startNumber to endNumber
    for (let i = startNumber; i <= endNumber; i++) {
        const newFileName = `${baseFileName}_${i}${extension}`;
        const newFilePath = folderPath + newFileName;
        imagePaths.push(newFilePath);
    }
    
    return imagePaths;
}

function getExpressionImages(expression) {
    const items = expression.split(/[ ,;]+/);
    const itemsNot = [];
    // Collect items that should be excluded
    items.forEach(item => {
        if (item[0] === '!') {
            itemsNot.push(item.substring(1));
        }
    });
    
    const result = [];
    // Filter imagePaths based on inclusion/exclusion criteria
    imagePaths.forEach(imagePath => {
        const isExcluded = itemsNot.some(substring => imagePath.includes(substring));
        const isIncluded = items.length - itemsNot.length === 0 || items.some(substring => imagePath.includes(substring));
        
        if (!isExcluded && isIncluded) {
            result.push(imagePath); // Push the imagePath instead of items
        }
    });
    return result; 
}