<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <title>Laravel</title>

        <!-- Fonts -->
        <link href="https://fonts.googleapis.com/css?family=Nunito:200,600" rel="stylesheet">

        <!-- Styles -->
        <style>
            html, body {
                background-color: #fff;
                color: #636b6f;
                font-family: 'Nunito', sans-serif;
                font-weight: 200;
                height: 100vh;
                margin: 0;
            }

            .full-height {
                height: 100vh;
            }

            .flex-center {
                align-items: center;
                display: flex;
                justify-content: center;
            }

            .position-ref {
                position: relative;
            }

            .top-right {
                position: absolute;
                right: 10px;
                top: 18px;
            }

            .content {
                text-align: center;
            }

            .title {
                font-size: 84px;
            }

            .links > a {
                color: #636b6f;
                padding: 0 25px;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: .1rem;
                text-decoration: none;
                text-transform: uppercase;
            }

            .m-b-md {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <div class="flex-center position-ref full-height">
            @if (Route::has('login'))
                <div class="top-right links">
                    @auth
                        <a href="{{ url('/home') }}">Home</a>
                    @else
                        <a href="{{ route('login') }}">Login</a>

                        @if (Route::has('register'))
                            <a href="{{ route('register') }}">Register</a>
                        @endif
                    @endauth
                </div>
            @endif

            <div class="content">
                <div class="row" style="width:100%">
                    <table>
                        <tr>
                            <td style="width:80%; text-align: left">
                                <video width="1280" height="720" autoplay muted controls>
                                    <source src="http://127.0.0.1:8080/go.ogg" type="video/ogg" onerror="reload()">
                                    Your browser does not support the video tag.
                                </video>
                            </td>
                            <td style="width:20%" id="history">
                                <table border="" style="text-align: center; width:100%">
                                    <thead>
                                        <th style="width: 25px;">Id</th>
                                        <th style="width: 25px;">Status</th>
                                    </thead>
                                    <tbody id="tab_history">
            
                                    </tbody>
                                </table>
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
        <script src="https://code.jquery.com/jquery-3.4.1.min.js" integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=" crossorigin="anonymous"></script>
        <script>
            let jsonData = new Array();
            const removeChildren = (node) => {
                while (node.firstChild) {
                    node.removeChild(node.firstChild);
                }
            }
            $(document).ready(function(){
                setInterval(function(){
                    $.ajax({
                        url : "{{route('ajax')}}",
                        type : "GET",
                        dataType : "json",
                        contentType: "application/json; charset=utf-8",
                        success: function(data){
                            jsonData = data;
                            console.log(jsonData);
                            let dataDiv = document.getElementById('tab_history');
                            let fragment = document.createDocumentFragment();
                            let i = 0;
                            for (const data of jsonData) {
                                let rows = document.createElement('tr');
                                let nama = document.createElement('td');
                                nama.setAttribute('class', 'Id');
                                nama.textContent = `${data.id}`;
                                let waktu = document.createElement('td');
                                waktu.setAttribute('class', 'Status');
                                waktu.textContent = `${data.nama}`;
                                fragment.appendChild(rows);
                                rows.appendChild(nama);
                                rows.appendChild(waktu);
                                i++;
                            }
                            removeChildren(dataDiv);
                            dataDiv.appendChild(fragment);
                        }
                    }, "json");
                }, 1000)
            })

            function reload()
            {
                location.reload()
            }
        </script>
    </body>
</html>