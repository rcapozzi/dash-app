// custom.js

function updateGraph(id, message) {
  var graphDiv = document.getElementById(id);
  var graphData = graphDiv.data;
  if (message === undefined) {
    console.log('updateGraph ! message undefined');
    return;
  }
  if (!('data' in message)) {
    console.error('updateGraph ! message undefined');
    return;
  }
  if (graphData === undefined) {
    console.log('updateGraph - graphData undefined. Creating.');
    graphData = [];
  }

  var ts = message.ts;
  var messageData = message.data;
  for (var i = 0; i < messageData.length; i++) {
    var symbol = messageData[i][0];
    var value = messageData[i][1];

    // Check if the symbol already exists in the graph data
    var existingData = graphData.find(function(item) {
      return item.name === symbol;
    });

    if (existingData) {
      // Symbol already exists, update x and y values
      existingData.x.push(ts);
      existingData.y.push(value);
    } else {
      // Symbol doesn't exist, create a new hash and append it to graph data
      var newData = {
        x: [ts],
        y: [value],
        name: symbol
      };
      graphData.push(newData);
    }
  }
  Plotly.react(graphDiv,graphData,{title:'v.' + ts, datarevision: ts})
  delete message['data'];
  return message;
}


// Function to update the graph with new data
function updateGraphDemo(id, newData) {
  // console.log('updateGraph <<', id, newData);
  var graphContainer = document.getElementById(id);
  if (newData === undefined){
    console.log('updateGraph >> newData undefined');
    return;
  }
  if (graphContainer.data !== undefined){
    // console.log('updateGraph -- graphContainer already defined');
    return;
  }
  console.log('updateGraph -- graphContainer has no data. Making it...', id, newData);

  var data = [{
    x: [1999, 2000, 2001, 2002],
    y: [10, 15, 13, 17],
    name: '1st',
    type: 'scatter'
  }];
  var updatedLayout = {
    title: 'Live Graph'
  };

  Plotly.react(graphContainer, data, updatedLayout);
  data[0].name = 'New Name';
  data.push({
    x: [1999, 2000, 2001, 2002],
    y: [20, 25, 23, 27],
    name: '2nd',
    type: 'scatter'
  });
  Plotly.react(graphContainer, data, updatedLayout);
  data[0].name = 'New Name2';
  data[1].x.push(2003);
  data[1].y.push(20);
  // Bump revision to trigger redraw.
  Plotly.react(graphContainer, data, { title: 'Some Title', datarevision: 0 });
}

// Function to fetch the incremental data from the server
function fetchData() {
  // Make an HTTP request to the server endpoint that provides the incremental data
  // You can use different methods like fetch(), XMLHttpRequest, etc.

  // Example using fetch() method
  fetch('/update?type=example')
    .then(function(response) {
      return response.json();
    })
    .then(function(data) {
      // Call the updateGraph() function with the received data
      updateGraph(data);
    })
    .catch(function(error) {
      console.error('Error fetching data:', error);
    });
}
// // Call the fetchData() function to initially fetch and update the graph
// fetchData();

// // Set an interval to periodically fetch and update the graph with incremental data
// setInterval(fetchData, 60000); // 60 seconds interval

function formatDateTime(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
};

function convertDates(series) {
  const easternTimezone = 'America/New_York';
  return series.map(value => {
    if (typeof value === 'number') {
      const timestamp = value < Math.pow(2, 32) ? value * 1000 : value;
      const date = new Date(timestamp);
      return formatDateTime(date);
    } else if (typeof value === 'string') {
      return value; // For string values, keep them as-is
    } else {
      return value; // For Date values, skip the conversion and keep them as-is
    }
  })
};

function graph1_cb(data) {
  // console.log('graph1_cb enter', data);
  if(data === undefined) {
      console.log('graph1_cb data is undefined');
      return {'data': [], 'layout': {}};
  }
  const y2 = data.y.map(value => -value);
  const fig = {
      'data': [data, {'x': data.x, 'y': y2}],
      'layout': {
          'title': 'Memory Gaph Client Side'
       }
  };
  return fig;
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
      graph1_cb: function(data) {
          return graph1_cb(data);
      },
      merge_stores: function(data, old_data) {
        if(data === undefined) {
          return {};
        }
        data.x = convertDates(data.x);
        if(old_data === undefined) {
          return data;
        }

        const out_data = {
          'x': [...old_data.x, ...data.x],
          'y': [...old_data.y, ...data.y]
        }
        // console.log('merge_stores return data', out_data);
        return out_data;
      }
  }
});


