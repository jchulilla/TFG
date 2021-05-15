<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sentiment Analysis</title>
  <link href="bootstrap/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
  <script src="https://code.jquery.com/jquery-3.5.0.js"></script>
  <script src="bootstrap/js/bootstrap.min.js"></script>  
<form action="index.php" method="POST" title>
  <link type="text/css" rel="stylesheet" property="stylesheet" href="css/newstyle.css">
  <div class="container-fluid">
  <div class="row">
  <div class="col-md-12 well">
  <label class="title">Sentiment Analysis</label>
  </div>
  </div>
  <div class="row">
  <div class="col-md-3 well">
  <div class="elem-group">
    <label for="classifier">Choose Classifier</label>
    <select id="classifier" name="classifier" required>
        <option value="LR" <?php if(@$_POST['classifier'] == 'LR') { echo 'selected = \"selected\"'; } ?>>Logistic Regression</option>
        <option value="MNB" <?php if(@$_POST['classifier'] == 'MNB') { echo 'selected = \"selected\"'; } ?>>Multinomial Naïve Bayes</option>
        <option value="ULM" <?php if(@$_POST['classifier'] == 'ULM') { echo 'selected = \"selected\"'; } ?>>ULMFiT</option>
    </select>
  </div>
  <br><br><br>
  <button type="submit" name="Apply">Apply</button>
  </div>
  <div class="col-md-8 well">
  <div class="elem-group">
      <label for="txt">Text</label>
      <textarea id="txt" name="txt" required placeholder="Enter your text"></textarea> <br><br><br>
      <label for="log">Console log</label>
      <textarea readonly><?php if(isset($_POST['Apply'])){$message = escapeshellarg ($_POST['txt']);
                                                          $model=$_POST['classifier'];
                                                          if ($model == 'LR') {
                                                            $output = shell_exec("python3 ./Python/TFG_jchulilla_PEC3_TEST_LR.py $message");}
                                                          else if ($model == 'ULM') {
                                                            $output = shell_exec("python3 ./Python/TFG_jchulilla_PEC3_TEST_ULM.py $message");}
                                                          else if ($model == 'MNB') {
                                                            $output = shell_exec("python3 ./Python/TFG_jchulilla_PEC3_TEST_MNB.py $message");}
                                                          else {$output = "ERROR";}
                                                          print_r($message);
                                                          echo("").PHP_EOL;
                                                          echo("").PHP_EOL;
                                                          print_r("Results.....");
                                                          echo("").PHP_EOL;
                                                          print_r($output);}
                                                          ?>
      </textarea>
  </div >
</div >
</div >
</div >
</form>
</body>
</html>
