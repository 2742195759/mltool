/* from the electron quick-start */
var electron = require('electron');
var ipcMain = require('electron').ipcMain;
var app = electron.app;
var mainwin = null;
var name2win = {}
var name2output = {}
var name2input = {}

function setCallback(){
    ipcMain.on('dialog-create', (event, args) => {
        name = args['name']
        input = args['input']
        newWin = new electron.BrowserWindow({
            parent: mainwin, 
            width:500,
            height:500,
            modal:true,
            frame:true,//是否显示边缘框
            fullscreen:false, //是否全屏显示,
            webPreferences: {
                nodeIntegration: true,
                enableRemoteModule: true
            },
        })
        newWin.loadURL(`file://${__dirname}/`+ name +'.html');
        name2win[name] = newWin
        name2input[name] = input
        newWin.on('close', ()=>{
            event.reply('dialog-output', name2output[name])
            newWin = null
            name2win[name] = null
        })
    })
    ipcMain.on('dialog-close', (event, args)=>{
        name = args['name']
        output = args['output']
        name2output[name] = output
        name2win[name].close()
    })
    ipcMain.on('dialog-input',(event, args)=>{
        name = args['name']
        input = name2input[name]
        event.returnValue = input
    })
}

function createWindow() {
	if (mainwin) return;
	win = new electron.BrowserWindow({
		width: 800, height: 600,
        fullscreen: true,
		webPreferences: {
			nodeIntegration: true,
			enableRemoteModule: true
		}
	});
	win.loadURL("file://" + __dirname + "/index.html");
	//win.webContents.openDevTools();
	win.on('closed', function () { win = null; });
    mainwin = win
    setCallback()
}
app.on('open-file', function () { console.log(arguments); });
app.on('ready', createWindow);
app.on('activate', createWindow);
app.on('window-all-closed', function () { if (process.platform !== 'darwin') app.quit(); });
