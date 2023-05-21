// custom.js

// Assuming you have included the Plotly.js library

// Function to update the graph with new data
function updateGraph(newData) {
  // Access the existing graph container
  var graphContainer = document.getElementById('live-graph1');

  // Update the graph data with the new data
  var updatedData = {
    x: newData.x,
    y: newData.y,
    type: 'line'
  };

  // Update the graph layout if necessary
  var updatedLayout = {
    title: 'Live Graph'
  };

  // Plotly.js update function to redraw the graph
  Plotly.update(graphContainer, [updatedData], updatedLayout);
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

function do_graph1(data) {
  console.log('do_graph1 enter', data);
  if(data === undefined) {
      console.log('do_graph1 data is undefined');
      return {'data': [], 'layout': {}};
  }
  const fig = {
      // 'data': [{'x': data.x, 'y': data.y}],
      'data': [data],
      'layout': {
          'title': 'Memory Gaph Client Side'
       }
  };
  return fig;
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
      do_graph1: function(data) {
          return do_graph1(data);
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
        console.log('merge_stores return data', out_data);
        return out_data;
      }
  }
});


