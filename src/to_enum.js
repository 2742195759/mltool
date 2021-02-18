page_name = 'to_enum'
const $ = require('jquery');
const ipcRenderer = require('electron').ipcRenderer
inputs = ipcRenderer.sendSync('dialog-input', {'name': page_name})

column_name = inputs['column']
column_values = inputs['values']

$('#paperInputs').attr('placeholder', column_name)
var grid = canvasDatagrid({
    parentNode: document.getElementById('grid'),
    borderDragBehavior: 'move',
    allowMovingSelection: true,
    columnHeaderClickBehavior: 'select',
    allowFreezingRows: true,
    allowFreezingColumns: true,
    allowRowReordering: true,
    tree: false,
    debug: false,
    showPerformance: false,
    editable: true,
})
grid.style.height = '100%';
grid.style.width = '100%';
data_tmp = []
column_values.forEach((v,i,a)=>{
    data_tmp.push({'value': v, 'int': i})
})
grid.data = data_tmp
$('#ok').click(()=>{
    // 获取 grid 的 data
    val2id = {}
    grid.data.forEach((v,i,a)=>{
        val2id[v['value']] = v['int']
    })
    ipcRenderer.send('dialog-close', {'name':page_name, 'output':val2id})
})
$('#no').click(()=>{
    ipcRenderer.send('dialog-close', {'name':page_name, 'output':null})
})
