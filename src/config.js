var fs = require('fs');
page_name = 'config'
const $ = require('jquery');
const ipcRenderer = require('electron').ipcRenderer
inputs = ipcRenderer.sendSync('dialog-input', {'name': page_name})

column_name = inputs['column_name']
column_values = inputs['values']
form_name = inputs['form_name']

// 流的方式读取文件
var readStream = fs.createReadStream(__dirname + '/' + form_name + '.html')
var form_str = '';
var count = 0; // 次数
readStream.on('data', function(chunk){
    form_str += chunk;
    count++;
})
readStream.on('end', function(chunk) {
    $('#formhtml')[0].innerHTML = form_str
    $('#formjs').attr('src', __dirname + '/' + form_name + '.js')
})
readStream.on('error', function(err) {
    console.log(err);
});

$('#columnlabel')[0].innerHTML = 'Selected Column: ' + `<font color="red">${column_name}</font>`
$('#ok').click(()=>{
    // 获取 grid 的 data
    output = form_gather_output(inputs) // 这个函数在 formjs 中被定义
    ipcRenderer.send('dialog-close', {'name':page_name, 'output':output})
})
$('#no').click(()=>{
    ipcRenderer.send('dialog-close', {'name':page_name, 'output':null})
})
