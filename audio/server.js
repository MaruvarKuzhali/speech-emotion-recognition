var express = require('express');
var bodyParser = require('body-parser');
var app     = express();

app.listen(3000, function() {
  console.log('Server running at http://127.0.0.1:3000/');
});

app.get('/', call); 

function call(req, res) { 
  var spawn = require("child_process").spawn; 
  var process = spawn('python',["print.py", res.query.audioplay]); 
  
  process.stdout.on('data', function(data) { 
        res.send(data.toString()); 
    } ) 
   
} 


