function testGmailExport() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName("emailgrabber"); 
  
  if (!sheet) {
    Logger.log("Error: Could not find 'emailgrabber'.");
    return;
  }
  
  // Grab the 10 most recent email threads
  var threads = GmailApp.search("in:anywhere", 0, 10);
  Logger.log("Found " + threads.length + " threads.");

  // 🎯 CORRECTED ENDPOINT ROUTE FOR GRADIO 4+ CUSTOM ENDPOINTS
  var apiUrl = "https://amandava284-spamdetector.hf.space/gradio_api/call/predict_email"; 

  for (var i = 0; i < threads.length; i++) {
    var messages = threads[i].getMessages();
    if (!messages || messages.length === 0) continue;
    
    var message = messages[0]; 
    var date = message.getDate();
    var from = message.getFrom();
    var subject = message.getSubject();
    var body = message.getPlainBody();

    var predictionResult = "⚠️ Classification Failed"; 
    
    if (body && body.trim() !== "") {
      var payload = { 
        "data": [body]
      };
      
      var options = {
        "method": "post",
        "contentType": "application/json",
        "payload": JSON.stringify(payload),
        "muteHttpExceptions": true 
      };
      
      try {
        var response = UrlFetchApp.fetch(apiUrl, options);
        var responseCode = response.getResponseCode();
        var responseText = response.getContentText();
        
        // Gradio 4+ /call/ endpoint returns a event-id json string: {"event_id": "..."}
        if (responseCode === 200) {
          var initialJson = JSON.parse(responseText);
          var eventId = initialJson.event_id;
          
          if (eventId) {
            // Step 2: Fetch the actual result from the assigned event cache
            var resultUrl = "https://amandava284-spamdetector.hf.space/gradio_api/call/predict_email/" + eventId;
            var resultResponse = UrlFetchApp.fetch(resultUrl, { "method": "get", "muteHttpExceptions": true });
            var resultText = resultResponse.getContentText();
            
            // Clean out the stream syntax 'data: [...]' to extract pure JSON arrays
            if (resultText.includes("data:")) {
              var cleanLines = resultText.split("\n");
              for (var k = 0; k < cleanLines.length; k++) {
                if (cleanLines[k].startsWith("data:")) {
                  var jsonStr = cleanLines[k].replace("data:", "").trim();
                  var parsedData = JSON.parse(jsonStr);
                  
                  var predictionData = parsedData[0];
                  if (typeof predictionData === 'object' && predictionData !== null) {
                    predictionResult = predictionData.label; 
                  } else {
                    predictionResult = predictionData;
                  }
                  break;
                }
              }
            }
          }
        } else if (responseCode === 404) {
          // Robust Fallback: Try the legacy /api routing with the proper prefix
          var fallbackUrl = "https://amandava284-spamdetector.hf.space/gradio_api/predict/predict_email";
          var fbResponse = UrlFetchApp.fetch(fallbackUrl, options);
          if (fbResponse.getResponseCode() === 200) {
            var fbJson = JSON.parse(fbResponse.getContentText());
            var fbData = fbJson.data[0];
            predictionResult = (typeof fbData === 'object') ? fbData.label : fbData;
          } else {
            predictionResult = "⚠️ Route Endpoint Mismatch";
          }
        } else {
          predictionResult = "❌ Error " + responseCode;
        }
      } catch (e) {
        predictionResult = "⚠️ Processing Failure";
      }
    } else {
      predictionResult = "❓ Empty Email Body";
    }

    // Append data securely to the bottom of the active rows
    sheet.appendRow([date, from, subject, body, predictionResult]);
  }
  
  Logger.log("Execution complete!");
}
