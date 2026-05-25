function checkAndBlockSenders() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var grabberSheet = ss.getSheetByName("emailgrabber");
  var blocklistSheet = ss.getSheetByName("blocklist");
  
  if (!grabberSheet || !blocklistSheet) {
    Logger.log("Error: Make sure 'emailgrabber' and 'blocklist' tabs both exist.");
    return;
  }
  
  var grabberData = grabberSheet.getDataRange().getValues();
  var blocklistData = blocklistSheet.getDataRange().getValues();
  
  // Create a list of senders we have already blocked to avoid duplicates
  var alreadyBlocked = [];
  for (var b = 1; b < blocklistData.length; b++) {
    if (blocklistData[b][0]) {
      alreadyBlocked.push(blocklistData[b][0].toLowerCase().trim());
    }
  }
  
  // Count how many spam emails each sender has sent
  var senderSpamCount = {};
  
  // Loop through your logs (skipping header row 0)
  for (var i = 1; i < grabberData.length; i++) {
    var rawFrom = grabberData[i][1]; // Column B: From
    var prediction = grabberData[i][4]; // Column E: Prediction Result
    
    if (!rawFrom || !prediction) continue;
    
    // Extract clean email address from "Name <email@domain.com>" format
    var emailMatch = rawFrom.match(/<([^>]+)>/) || [null, rawFrom];
    var cleanEmail = emailMatch[1].toLowerCase().trim();
    
    // Check if the model marked it as Spam
    if (prediction.toUpperCase().includes("SPAM")) {
      senderSpamCount[cleanEmail] = (senderSpamCount[cleanEmail] || 0) + 1;
    }
  }
  
  // Track new blocks to process at the end
  var newBlocksCount = 0;
  
  // Analyze counts and block if they hit the threshold (3 or more)
  for (var sender in senderSpamCount) {
    if (senderSpamCount[sender] >= 3) {
      
      // If they aren't already on our blocklist, add them
      if (alreadyBlocked.indexOf(sender) === -1) {
        blocklistSheet.appendRow([sender, new Date()]);
        Logger.log("🚫 Added to Blocklist: " + sender + " (Sent " + senderSpamCount[sender] + " spam emails)");
        newBlocksCount++;
        
        // Execute the block instantly by moving their current emails to trash
        try {
          var threadsToTrash = GmailApp.search("from:" + sender);
          for (var t = 0; t < threadsToTrash.length; t++) {
            threadsToTrash[t].moveToTrash();
          }
          Logger.log("🗑️ Moved existing threads from " + sender + " to Trash.");
        } catch (gmailError) {
          Logger.log("Error trashing emails for " + sender + ": " + gmailError.toString());
        }
      }
    }
  }
  
  if (newBlocksCount === 0) {
    Logger.log("Check complete. No new repeat spammers found.");
  }
}
