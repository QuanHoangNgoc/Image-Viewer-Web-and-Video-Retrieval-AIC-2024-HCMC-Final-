const BACK = '\\'; 
const EXPORT_FILE = "export.csv"

//!!! PLEASE replace config 
function createImagePath(baseName, frameId) { 
    const baseVideo = baseName.substring(0, baseName.lastIndexOf('_'));
    const number = (frameId / 25) + 1; 
    return `D:\\cd_data_C\\Desktop\\Web\\Images\\AIC2024KeyFrames\\Keyframes_${baseVideo}\\${baseName}\\${baseName}_${number}.jpg`;
}


function mappingFrame(num) {
    return (num-1) * 25;   
}

function mappingTextResult(filePath) {
    const baseFileName = filePath.substring(filePath.lastIndexOf(BACK) + 1, filePath.lastIndexOf('_'));
    const numFileName = filePath.substring(filePath.lastIndexOf('_') + 1, filePath.lastIndexOf('.')); 
    let num = parseInt(numFileName, 10);
    num = mappingFrame(num); 
    const newText = `${baseFileName},${num}`;
    return newText; 
}

function changeMappingTextBox(filePath) {
    const newText = mappingTextResult(filePath); 
    
    document.getElementById('mappingTextBox').value = newText;
}

