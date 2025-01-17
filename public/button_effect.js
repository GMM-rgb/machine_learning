const clearChatButton = document.getElementById('clear-button');
const saveButtonC = document.getElementById('save-button');
const uploadButton = document.getElementById('load-button');

const effectWait1 = '125';
const effectWait2 = '450';

function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
clearChatButton.addEventListener('mouseover', pushButtonDownVisual => {
    if(clearChatButton.style.boxShadow === '0 4.25px 0px rgb(100, 100, 100)' || clearChatButton.style.boxShadow === '' || clearChatButton.style.transform === 'translateY(0px)' || clearChatButton.style.transform === '') {
        clearChatButton.style.boxShadow = '0 2.25px 0px rgb(100, 100, 100)';
        clearChatButton.style.transform = 'translateY(2px)';
    wait(effectWait1).then(() => {
        clearChatButton.style.border = '3px solid rgb(0, 234, 255)';
        wait(effectWait2).then(() => {
            clearChatButton.style.borderTop = '3px solid rgb(179, 179, 180)';
            clearChatButton.style.borderBottom = '3px solid rgb(179, 179, 180)';
            clearChatButton.style.borderLeft = '3px solid rgb(179, 179, 180)';
            clearChatButton.style.borderRight = '3px solid rgb(179, 179, 180)';
        });
    });
    } else {
        return;
    }
});
clearChatButton.addEventListener('mouseout', revertPushButtonDownVisual => {
    if(clearChatButton.style.boxShadow === '0 2.25px 0px rgb(100, 100 ,100)' || clearChatButton.style.boxShadow === '' || clearChatButton.style.transform === 'translateY(2px)' || clearChatButton.style.transform === '') {
        clearChatButton.style.boxShadow = '0 4.25px 0px rgb(100, 100, 100)';
        clearChatButton.style.transform = 'translateY(0px)';
    } else {
        return;
    }
});

function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
clearChatButton.addEventListener('click', highlightButton => {
    clearChatButton.style.border = '3px solid rgb(0, 234, 255)';
    if(clearChatButton.style.boxShadow === '0 2.25px 0px rgb(100, 100, 100)' || clearChatButton.style.boxShadow === '' || clearChatButton.style.transform === 'translateY(3px)' || clearChatButton.style.transform === '') {
        clearChatButton.style.boxShadow = '0 1.25px 0px rgb(100, 100, 100)';
    }
});
