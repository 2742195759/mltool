var process = require('child_process');
const BrowserWindow =require('electron').remote.BrowserWindow
var parent_win = require('electron').remote.getCurrentWindow()
const ipcRenderer = require('electron').ipcRenderer
var $ = require('jquery')
var g_grid=undefined;   //canvasDatagrid() object
var g_ylabel=undefined; // str : head name 
var g_status = {
    'grid'  : undefined, 
    'ylabel' : undefined,  // str
}

function startDialog(args, func) // args.keys = ['page_name', 'form_name', 'column_name'] 
{
    ipcRenderer.once('dialog-output', (event, args)=>{
        func(event, args)
    })
    ipcRenderer.send('dialog-create', {'name':args.page_name,'input':args})
}

function startFormDialog(args, func) // args.keys = ['form_name', 'column_name'] 
{
    args.page_name = 'config'
    ipcRenderer.once('dialog-output', (event, args)=>{
        func(event, args)
    })
    ipcRenderer.send('dialog-create', {'name':args.page_name,'input':args})
}

function getPythonCmd(filename, args){
    args_str = " "
    for (var key in args){
        var val_str = args[key]
        if (typeof(val_str)=='object') {
            val_str = JSON.stringify(args[key])
        }
        args_str += ` --${key} \'${val_str}\' ` 
    }
    return 'python3 ' +  __dirname + '/../py_src/' + filename + args_str
}

function setGridStatus(stat){
    function setYlabel(new_value){
        if (g_ylabel){
            hd = g_grid.getHeaderByName(g_ylabel)
            hd.title = hd.name
            g_ylabel = undefined
        }
        g_ylabel = new_value
        hd = g_grid.getHeaderByName(g_ylabel)
        hd.title+='(Y)';
    }
    function setGrid(new_grid){
        g_grid = new_grid
    }
    if (stat['grid']) { // 先设置 grid
        setGrid(stat['grid'])
        g_status['grid'] = stat['grid']
    }
    if (stat['ylabel']){
        setYlabel(stat['ylabel'])
        g_status.yname = stat['ylabel']
    }
}

function python_installDependences(){
    process.exec('pip3 install numpy -i https://mirrors.aliyun.com/pypi/simple && ' + 
                 'pip3 install pandas -i https://mirrors.aliyun.com/pypi/simple && '+
                 'pip3 install sklearn -i https://mirrors.aliyun.com/pypi/simple && '+
                 'pip3 install matplotlib -i https://mirrors.aliyun.com/pypi/simple && '+
                 'pip3 install imblearn -i https://mirrors.aliyun.com/pypi/simple && '+
                 ''
              , (error, stdout, stderr) => {
    });
}

function python_loadDataSet(grid, filepath) {
    process.exec('cp ' + filepath + ` ${__dirname}/../cache/__tmp.csv`, (error, stdout, stderr) => {
        cmd = getPythonCmd('loadcsv.py', {'input':`${__dirname}/../cache/__tmp.csv`})
        console.log(cmd)
        process.exec(cmd, (error, stdout, stderr) => {
            grid.data=JSON.parse(stdout);
            setGridStatus(g_status) // 恢复状态
        });
    });
};

function getXArray(grid, ylabel){
    var xlabels = []
    grid.schema.forEach((v,i,a)=>{
        hd = v 
        if (hd.name != ylabel && hd.hidden != true) {
            xlabels.push(hd.name)
        }
    })
    return xlabels;
}

function python_deleteHidden(grid) {
    var xlabels = getXArray(grid, undefined)
    cmd = getPythonCmd('savecsv.py', {'input':`${__dirname}/../cache/__tmp.csv`, 'output':`${__dirname}/../cache/__tmp.csv`, 'xlabels': xlabels})
    console.log("执行:" + cmd)
    process.exec(cmd, (error, stdout, stderr) => {
        grid.data=JSON.parse(stdout);
        setGridStatus(g_status) // 恢复状态
    });
};

function python_toEnum(grid, val2id, column) {
    cmd = getPythonCmd('to_enum.py', {'input':`${__dirname}/../cache/__tmp.csv`, 'output':`${__dirname}/../cache/__tmp.csv`, 'dict': val2id, 
                        'column':column})
    console.log("执行:" + cmd)
    process.exec(cmd, (error, stdout, stderr) => {
        grid.data=JSON.parse(stdout);
        setGridStatus(g_status) // 恢复状态
    });
};

function python_predict(grid, ylabel, ml_method){
    var xlabels = getXArray(grid, ylabel)
    var str_ylabel  = ylabel
    //cmd = 'python3 ./py_src/predict.py --input=./cache/__tmp.csv --output=./cache/__tmp.csv --ylabel '+str_ylabel+' --xlabels \''+str_xlabels+"\' --ml-method "+ml_method
    cmd = getPythonCmd('predict.py', {'input':`${__dirname}/../cache/__tmp.csv`, 'output':`${__dirname}/../cache/__tmp.csv`, 'xlabels': xlabels, 
                        'ml-method':ml_method, 'ylabel':str_ylabel})
    console.log("执行:" + cmd)
    process.exec(cmd , (error, stdout, stderr) => {
        if (!error){
            alert (stdout)
        }
        python_loadDataSet(grid, "./cache/__tmp.csv")
    });
}

function main() {
    function initGrid() {
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
            editable: false
        });
        grid.style.height = '100%';
        grid.style.width = '100%';
        grid.addEventListener('contextmenu', function (e) {
            current_header = e.cell.header
            e.items.push({
                title: '设置为Y',
                click: function (e) {
                    new_value = current_header.name
                    setGridStatus({'ylabel': new_value})
                }
            });
            e.items.push({
                title: '表格处理函数',
                items: [
                    {
                        title:  '删除Hidden列', 
                        click:  function(){
                            python_deleteHidden(g_grid) ; 
                        }, 
                    }, 
                ]
            });
            e.items.push({
                title: '预处理',
                items: [
                    {
                        title:  'normalization', 
                        click:  function(e){
                            page_name = 'to_enum'
                        }, 
                    }, 
                    {
                        title:  'one-hot 表示', 
                        click:  function(){},
                    },
                    {
                        title:  'Catogory化', 
                        click:  function(e){
                            header_name = current_header.name
                            function unique (arr) {
                                return Array.from(new Set(arr))
                            }
                            column_value = []
                            g_grid.data.forEach((v,i,a)=>{
                                column_value.push(v[header_name])
                            })
                            column_set = unique(column_value)
                            args = {
                                'page_name': 'to_enum',
                                'column_name': header_name,
                                'values': column_set,
                            }
                            startDialog(args, (e, args)=>{
                                output = args
                                if (output){
                                    alert(JSON.stringify(output))
                                    python_toEnum(g_grid, output, header_name)
                                }
                            }); 
                        },
                    },
                ]
            });
            e.items.push({
                title: '机器学习-分类预测',
                items: [
                    {
                        title:  'LogisticRegression(逻辑回归)', 
                        click:  function(){
                            python_predict(g_grid, g_ylabel, 'LogisticRegression')
                        }, 
                    }, 
                    {
                        title:  'DNN / MLP (神经网络)', 
                        click:  function(){
                            args = {
                                'form_name': 'form_mlp',
                                'column_name': current_header.name,
                            }
                            startFormDialog(args, (e,args)=>{
                                console.log(args)
                                python_predict(g_grid, g_ylabel, 'MLP')
                            })
                        }, 
                    }, 
                    {
                        title:  'Naive Bayes(朴素贝叶斯)', 
                        click:  function(){
                            python_predict(g_grid, g_ylabel, 'NaiveBayes')
                        }, 
                    },
                    {
                        title:  'Decision Tree(决策树)', 
                        click:  function(){
                            python_predict(g_grid, g_ylabel, 'DecisionTree')
                        }, 
                    },
                ]
            });
        });
        grid.addEventListener('rendercell', function(e){  //添加颜色 X Y 设置为不同的颜色..  Y 设置为绿色
            if (/\(Y\)/.test(e.cell.header.title)){
                e.ctx.fillStyle = '#AEEDCF';
            }
            if (/output_/.test(e.cell.header.title)){
                e.ctx.fillStyle = '#EEECCC';
            }
        });
        return grid; 
    };
    $('#readfile').click(function (){
        $('#readfile').hide()
        $('#inputfile').hide()
        var inputfile = $('#inputfile')[0].files[0].path
        setGridStatus({'grid': initGrid()})
        python_loadDataSet(g_grid, inputfile)
    }); 
}

if (document.addEventListener) {
    document.addEventListener('DOMContentLoaded', main);
} else {
    setTimeout(function () {
        'use strict';
        main();
    }, 500);
}
