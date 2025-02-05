document.addEventListener('DOMContentLoaded', () => {
    let settings = document.getElementById('settings-menu-button');
    let settingsMenu = document.getElementById('settings-menu');
    let settingsMenuClose = document.getElementById('settings-menu-close-button');

    settings.addEventListener('click', () => {
        settingsMenu.style.display = 'flex';
    });

    settingsMenuClose.addEventListener('click', () => {
        settingsMenu.style.display = 'none';
    });
});
