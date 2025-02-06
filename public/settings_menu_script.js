document.addEventListener('DOMContentLoaded', () => {
    let settings = document.getElementById('settings-menu-button');
    let settingsMenu = document.getElementById('settings-menu');
    let settingsMenuClose = document.getElementById('settings-menu-close-button');

    settings.addEventListener('click', () => {
        if(settingsMenu.style.display === 'none' || settingsMenu.style.display === '') {
            settingsMenu.style.display = 'flex';
            return;
        } else if(settingsMenu.style.display === 'flex' || settingsMenu.style.display === '') {
            settingsMenu.style.display = 'none';
            return;
        }
    });

    settingsMenuClose.addEventListener('click', () => {
        settingsMenu.style.display = 'none';
    });
    let settingsMenuItems = document.querySelectorAll('.settings-menu-item');
    settingsMenuItems.forEach((item) => {
        item.addEventListener('click', () => {
            settingsMenu.style.display = 'none';
        });
    });
});
